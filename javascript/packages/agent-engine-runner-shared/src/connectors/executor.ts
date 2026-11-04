/**
 * Bounded HTTP execution for compiled connector operations.
 *
 * One executor owns the transport for one connector in one Tool Pod process.
 * It never registers tools or reads credentials at construction time. The
 * request path validates arguments and serializes wire values before any I/O,
 * then applies the catalog's static or delegated auth, follows only
 * same-origin redirects, and retains only a bounded error envelope.
 *
 * Mirrors `agent_engine_runner_shared/connectors/executor.py`.
 */

import { randomUUID } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { PassThrough, type Readable, type Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { rootCertificates } from "node:tls";
import { createBrotliDecompress, createGunzip, createInflate } from "node:zlib";

import { Ajv2020 } from "ajv/dist/2020.js";
import { Agent, request as undiciRequest } from "undici";

import {
  isRecord,
  validateSegment,
  validateUrl,
  type Operation,
  type ToolDefinitions,
} from "./definitions.js";
import type { ToolAuthorization } from "../models.js";
import { tenantEnvVars } from "../utils.js";

export const DEFAULT_EXECUTION_TIMEOUT_SECONDS = 30;
export const MAX_RESPONSE_BYTES = 200_000;
export const MAX_REDIRECTS = 5;
export const MAX_RETRIES = 3;
export const MAX_ERROR_ENVELOPE_BYTES = 4_096;
export const USER_AGENT = "agent-engine-connector";

const RETRYABLE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);
const AJV = new Ajv2020({
  strict: false,
  validateFormats: false,
  allErrors: false,
  // The catalog compiler accepts identity escapes in generated patterns.
  unicodeRegExp: false,
});

/** One request handed to a transport, after validation and serialization. */
export interface ConnectorTransportRequest {
  readonly method: string;
  readonly url: string;
  readonly headers: Readonly<Record<string, string>>;
  readonly body?: Uint8Array;
  readonly signal: AbortSignal;
}

/** The subset of a response the executor needs from a transport. */
export interface ConnectorTransportResponse {
  readonly status: number;
  readonly headers: { get(name: string): string | null };
  readonly body: AsyncIterable<Uint8Array> & {
    dump?(): Promise<void>;
    destroy?(): void;
    cancel?(): Promise<void>;
  };
}

/** One HTTP hop. Tests inject a handler; production uses undici. */
export type ConnectorTransport = (
  request: ConnectorTransportRequest,
) => Promise<ConnectorTransportResponse>;

export interface ConnectorExecutorOptions {
  readonly transport?: ConnectorTransport;
  readonly timeoutSeconds?: number;
  readonly now?: () => number;
  readonly sleep?: (milliseconds: number) => Promise<void>;
}

export class ConnectorHTTPError extends Error {
  readonly status: number;
  readonly response: { status: number; data?: unknown };

  constructor(status: number, data?: unknown) {
    super(`connector API returned HTTP ${status}`);
    this.name = "ConnectorHTTPError";
    this.status = status;
    this.response = data === undefined ? { status } : { status, data };
  }
}

class ConnectorTimeoutError extends Error {
  constructor() {
    super("connector request timed out");
    this.name = "TimeoutError";
  }
}

/**
 * A transport or response-decoding failure with the code the shared classifier
 * uses for CONNECTION_ERROR. The provider's original message is deliberately
 * dropped; it can contain request details the user must not see.
 */
class ConnectorTransportError extends Error {
  readonly code = "UND_ERR_SOCKET";

  constructor(message: string) {
    super(message);
    this.name = "ConnectorTransportError";
  }
}

const TIMEOUT_CODES = new Set([
  "ETIMEDOUT",
  "UND_ERR_HEADERS_TIMEOUT",
  "UND_ERR_BODY_TIMEOUT",
  "UND_ERR_CONNECT_TIMEOUT",
]);

function isTimeoutFailure(error: unknown): boolean {
  let current: unknown = error;
  const seen = new Set<unknown>();
  for (let depth = 0; current != null && depth < 5; depth += 1) {
    if (typeof current !== "object" || seen.has(current)) break;
    seen.add(current);
    const node = current as { name?: unknown; code?: unknown; cause?: unknown };
    if (
      current instanceof ConnectorTimeoutError ||
      node.name === "TimeoutError"
    ) {
      return true;
    }
    if (typeof node.code === "string" && TIMEOUT_CODES.has(node.code)) {
      return true;
    }
    current = node.cause;
  }
  return false;
}

function assertJsonValue(value: unknown, seen = new Set<object>()): void {
  if (value === null) return;
  if (typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error("arguments must be JSON values");
    }
    return;
  }
  if (typeof value !== "object") {
    throw new Error("arguments must be JSON values");
  }
  if (seen.has(value)) throw new Error("arguments must be JSON values");
  seen.add(value);
  if (Array.isArray(value)) {
    for (const item of value) assertJsonValue(item, seen);
  } else {
    for (const item of Object.values(value)) assertJsonValue(item, seen);
  }
  seen.delete(value);
}

function quote(value: string, plusSpace: boolean): string {
  let encoded = encodeURIComponent(value).replace(
    /[!'()*]/g,
    (character) => `%${character.charCodeAt(0).toString(16).toUpperCase()}`,
  );
  if (plusSpace) encoded = encoded.replace(/%20/g, "+");
  return encoded;
}

function originOf(url: URL): string {
  return `${url.protocol}//${url.hostname}:${url.port}`;
}

function retryDelay(retryAfter: string, attempt: number): number {
  const delay = Math.min(2 ** (attempt + 1), 8);
  if (/^\d+$/.test(retryAfter)) {
    const seconds = Number.parseInt(
      retryAfter.replace(/^0+/, "").slice(0, 3) || "0",
      10,
    );
    return Math.max(delay, Math.min(seconds, 30));
  }
  if (retryAfter) {
    const timestamp = Date.parse(retryAfter);
    if (Number.isFinite(timestamp)) {
      const seconds = (timestamp - Date.now()) / 1000;
      return Math.max(delay, Math.min(seconds, 30));
    }
  }
  return delay;
}

function connectorCa(): string[] | undefined {
  const file = process.env["SSL_CERT_FILE"];
  const directory = process.env["SSL_CERT_DIR"];
  if (!file && !directory) return undefined;

  const custom: string[] = [];
  if (file) custom.push(readFileSync(file, "utf8"));
  if (directory) {
    for (const entry of readdirSync(directory)) {
      try {
        custom.push(readFileSync(join(directory, entry), "utf8"));
      } catch (error) {
        // Directory entries are hashed certificates on OpenSSL layouts. Skip
        // anything that is not a readable PEM file; a missing or unreadable
        // SSL_CERT_FILE still fails closed above.
        if ((error as NodeJS.ErrnoException).code !== "EISDIR") throw error;
      }
    }
  }
  return [...rootCertificates, ...custom];
}

function createAgent(): Agent {
  const ca = connectorCa();
  const options: ConstructorParameters<typeof Agent>[0] = {
    keepAliveTimeout: 60_000,
    keepAliveMaxTimeout: 600_000,
  };
  if (ca !== undefined) options.connect = { ca };
  return new Agent(options);
}

type UndiciRequestOptions = NonNullable<Parameters<typeof undiciRequest>[1]>;

function decodersForEncoding(encoding: string | null): Transform[] {
  const decoders: Transform[] = [];
  for (const value of (encoding ?? "").split(",")) {
    switch (value.trim().toLowerCase()) {
      case "gzip":
        decoders.push(createGunzip());
        break;
      case "deflate":
        decoders.push(createInflate());
        break;
      case "br":
        decoders.push(createBrotliDecompress());
        break;
      default:
        // `identity` and unknown codings need no decoder. HTTPX ignores
        // unknown entries the same way.
        break;
    }
  }
  return decoders;
}

function defaultTransport(agent: Agent): ConnectorTransport {
  return async (request) => {
    const options: UndiciRequestOptions = {
      method: request.method as UndiciRequestOptions["method"],
      headers: request.headers,
      signal: request.signal,
      dispatcher: agent,
      maxRedirections: 0,
    };
    if (request.body !== undefined) options.body = request.body;
    const response = await undiciRequest(request.url, options);
    // undici.request exposes raw response bytes and permits the GET/HEAD
    // bodies the compiled catalog contract allows. Decode content encodings
    // here so the response cap and returned text are measured after decoding,
    // matching the Python client.
    let body: ConnectorTransportResponse["body"] = response.body;
    // undici's public request typings still describe the runtime Headers
    // object as IncomingHttpHeaders. The runtime object has the Headers API
    // the executor needs (notably case-insensitive get()).
    const headers = new Headers(response.headers as unknown as HeadersInit);
    const bodyless =
      request.method === "HEAD" ||
      response.statusCode === 204 ||
      response.statusCode === 304;
    const decoders = bodyless
      ? []
      : decodersForEncoding(headers.get("content-encoding")).reverse();
    if (decoders.length > 0) {
      // Pipeline propagates teardown in both directions: truncating or
      // failing the decoded stream destroys the upstream HTTP response and
      // every intermediate decoder instead of leaving them open until the
      // invocation deadline.
      const decoded = new PassThrough();
      const chain = decoders as [Transform, ...Transform[]];
      void pipeline(
        response.body as unknown as Readable,
        ...chain,
        decoded,
      ).catch((error: unknown) => {
        if (!decoded.destroyed) {
          decoded.destroy(
            error instanceof Error
              ? error
              : new Error("connector response decoding failed"),
          );
        }
      });
      body = decoded as unknown as ConnectorTransportResponse["body"];
    }
    return {
      status: response.statusCode,
      headers,
      body,
    };
  };
}

async function discardBody(
  body: ConnectorTransportResponse["body"],
): Promise<void> {
  try {
    if (typeof body.cancel === "function") {
      await body.cancel();
    } else if (typeof body.dump === "function") {
      await body.dump();
    } else if (typeof body.destroy === "function") {
      body.destroy();
    } else {
      for await (const chunk of body) {
        // Drain so a pooled connection can be reused.
        void chunk;
      }
    }
  } catch {
    // Discarding a failed/redirected response is best effort.
  }
}

async function readBoundedBody(
  body: ConnectorTransportResponse["body"],
  deadline: number,
  now: () => number,
  signal: AbortSignal,
): Promise<string> {
  const chunks: Uint8Array[] = [];
  let total = 0;
  let truncated = false;
  try {
    for await (const chunk of body) {
      if (signal.aborted || now() >= deadline)
        throw new ConnectorTimeoutError();
      const room = MAX_RESPONSE_BYTES - total;
      if (chunk.byteLength > room) {
        chunks.push(chunk.subarray(0, room));
        total += room;
        truncated = true;
        break;
      }
      chunks.push(chunk);
      total += chunk.byteLength;
    }
  } catch (error) {
    if (isTimeoutFailure(error) || signal.aborted || now() >= deadline) {
      throw new ConnectorTimeoutError();
    }
    throw new ConnectorTransportError("connector response decoding failed");
  }
  if (truncated) await discardBody(body);
  if (signal.aborted || now() >= deadline) throw new ConnectorTimeoutError();
  const text = Buffer.concat(chunks).toString("utf8");
  return truncated ? `${text}\n...[truncated]` : text;
}

/** Execute one compiled connector operation against its configured origin. */
export class ConnectorExecutor {
  private readonly definitions: ToolDefinitions;
  private readonly operations: Map<string, Operation>;
  private readonly validators: Map<string, (value: unknown) => boolean>;
  private readonly agent: Agent | null;
  private readonly transport: ConnectorTransport;
  private readonly timeoutSeconds: number;
  private readonly now: () => number;
  private readonly sleep: (milliseconds: number) => Promise<void>;
  private closed = false;

  constructor(
    definitions: ToolDefinitions,
    options: ConnectorExecutorOptions = {},
  ) {
    const timeoutSeconds =
      options.timeoutSeconds ?? DEFAULT_EXECUTION_TIMEOUT_SECONDS;
    if (!Number.isFinite(timeoutSeconds) || timeoutSeconds <= 0) {
      throw new Error("connector timeout must be a positive finite number");
    }
    if (
      /\{[^{}]+\}/.test(definitions.source.base_url) &&
      !(definitions.auth?.api_keys ?? []).some(
        (credential) => credential.location === "path",
      )
    ) {
      throw new Error(
        "connector path credential must be materialized before execution",
      );
    }
    this.definitions = structuredClone(definitions);
    this.operations = new Map(
      this.definitions.tools.map((tool) => [tool.name, tool]),
    );
    this.validators = new Map(
      this.definitions.tools.map((tool) => [
        tool.name,
        AJV.compile(tool.inputSchema) as (value: unknown) => boolean,
      ]),
    );
    this.timeoutSeconds = timeoutSeconds;
    this.now = options.now ?? (() => performance.now());
    this.sleep =
      options.sleep ??
      ((milliseconds) =>
        new Promise<void>((resolve) => {
          setTimeout(resolve, milliseconds);
        }));

    if (options.transport !== undefined) {
      this.agent = null;
      this.transport = options.transport;
    } else {
      this.agent = createAgent();
      this.transport = defaultTransport(this.agent);
    }
  }

  /** Release pooled connections after in-flight calls have finished. */
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    await this.agent?.close();
  }

  /**
   * Execute a named operation and return bounded UTF-8 response text.
   *
   * Invalid definitions, arguments, secrets, and redirects fail before a
   * useful response is surfaced. Only GET/HEAD/OPTIONS retry 429/503.
   */
  async execute(
    name: string,
    arguments_: Record<string, unknown>,
    {
      env,
      authorization,
    }: {
      env?: Readonly<Record<string, string>>;
      authorization?: ToolAuthorization | null;
    } = {},
  ): Promise<string> {
    if (this.closed) throw new Error("connector executor is closed");
    const tool = this.operations.get(name);
    if (tool === undefined) throw new Error("unknown connector tool");

    const deadline = this.now() + this.timeoutSeconds * 1000;
    let prepared: {
      url: string;
      headers: Record<string, string>;
      body: Uint8Array | undefined;
      authQuery: Array<[string, string]>;
      hasPathAuth: boolean;
    };
    try {
      prepared = this._prepareRequest(tool, arguments_, env, authorization);
    } catch (error) {
      if (error instanceof RangeError) {
        throw new Error("arguments are too deeply nested", { cause: error });
      }
      throw error;
    }

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt += 1) {
      const result = await this._request(
        tool.method,
        prepared.url,
        prepared.headers,
        prepared.body,
        deadline,
        prepared.authQuery,
        prepared.hasPathAuth,
      );
      if (result.status >= 200 && result.status < 300) return result.text;
      if (
        (result.status === 429 || result.status === 503) &&
        RETRYABLE_METHODS.has(tool.method) &&
        attempt < MAX_RETRIES
      ) {
        const delay = retryDelay(
          result.headers.get("retry-after") ?? "",
          attempt,
        );
        if (delay >= this._remainingSeconds(deadline)) {
          throw new ConnectorTimeoutError();
        }
        await this.sleep(delay * 1000);
        continue;
      }
      throw this._httpError(result.status, result.text);
    }
    throw new Error("retry loop exhausted without a result");
  }

  private _prepareRequest(
    tool: Operation,
    arguments_: Record<string, unknown>,
    env: Readonly<Record<string, string>> | undefined,
    authorization: ToolAuthorization | null | undefined,
  ): {
    url: string;
    headers: Record<string, string>;
    body: Uint8Array | undefined;
    authQuery: Array<[string, string]>;
    hasPathAuth: boolean;
  } {
    if (!isRecord(arguments_)) {
      throw new Error("arguments do not match the compiled input schema");
    }
    const properties = isRecord(tool.inputSchema["properties"])
      ? tool.inputSchema["properties"]
      : {};
    const filtered = Object.fromEntries(
      Object.keys(properties)
        .filter((key) => key in arguments_)
        .map((key) => [key, arguments_[key]]),
    );
    const validate = this.validators.get(tool.name);
    if (validate === undefined || !validate(filtered)) {
      throw new Error("arguments do not match the compiled input schema");
    }
    assertJsonValue(filtered);

    let path = tool.path;
    for (const key of tool.params.path) {
      const value = this._paramValue(filtered, key);
      const rendered = Array.isArray(value)
        ? value.map((item) => quote(validateSegment(item), false)).join(",")
        : quote(validateSegment(value), false);
      path = path.split(`{${key}}`).join(rendered);
    }

    const query: Array<[string, string]> = [];
    for (const key of tool.params.query) {
      if (!(key in filtered)) continue;
      const raw = filtered[key];
      const encoding = tool.params.query_style[key] ?? "repeat";
      if (encoding === "json") {
        query.push([key, JSON.stringify(raw)]);
        continue;
      }
      if (raw === null || raw === undefined) continue;
      if (isRecord(raw)) {
        query.push(...this._bracketPairs(key, raw));
        continue;
      }
      const value = this._paramValue(filtered, key);
      if (Array.isArray(value) && encoding === "comma") {
        query.push([key, value.join(",")]);
      } else if (Array.isArray(value) && encoding === "brackets") {
        for (const item of value) query.push([`${key}[]`, item]);
      } else if (Array.isArray(value)) {
        for (const item of value) query.push([key, item]);
      } else {
        query.push([key, value]);
      }
    }

    const headers: Record<string, string> = {};
    for (const key of tool.params.header) {
      if (!(key in filtered) || filtered[key] === null) continue;
      const value = this._paramValue(filtered, key);
      headers[key] = Array.isArray(value) ? value.join(",") : value;
    }
    if (tool.params.cookie.length > 0) {
      const pairs = tool.params.cookie
        .filter((key) => key in filtered && filtered[key] !== null)
        .map((key) => `${key}=${quote(this._scalar(filtered[key]), false)}`);
      if (pairs.length > 0) headers["Cookie"] = pairs.join("; ");
    }

    const auth = this._applyAuth(headers, query, env, authorization);
    const body = this._requestBody(tool, filtered);
    if (body.contentType !== undefined) {
      headers["Content-Type"] = body.contentType;
    }
    if (
      !Object.keys(headers).some((name) => name.toLowerCase() === "user-agent")
    ) {
      headers["User-Agent"] = USER_AGENT;
    }

    const url = new URL(`${auth.baseUrl}${path}`);
    if (query.length > 0) {
      url.search = this._serializeQuery(query);
    }
    return {
      url: url.toString(),
      headers,
      body: body.bytes,
      authQuery: auth.authQuery,
      hasPathAuth: auth.hasPathAuth,
    };
  }

  private _applyAuth(
    headers: Record<string, string>,
    query: Array<[string, string]>,
    env: Readonly<Record<string, string>> | undefined,
    authorization: ToolAuthorization | null | undefined,
  ): {
    baseUrl: string;
    authQuery: Array<[string, string]>;
    hasPathAuth: boolean;
  } {
    const auth = this.definitions.auth;
    let baseUrl = this.definitions.source.base_url.replace(/\/+$/, "");
    if (auth === null || auth === undefined) {
      return { baseUrl, authQuery: [], hasPathAuth: false };
    }
    if (auth.provider != null) {
      const token = authorization?.token;
      if (!token || !isDelegatedToken(token)) {
        throw new Error("delegated connector token is missing or invalid");
      }
      headers["Authorization"] = `Bearer ${token}`;
      Object.assign(headers, auth.headers ?? {});
      return { baseUrl, authQuery: [], hasPathAuth: false };
    }

    const secrets = env ?? tenantEnvVars();
    const authQuery: Array<[string, string]> = [];
    let hasPathAuth = false;
    if (auth.bearer != null) {
      headers["Authorization"] = `Bearer ${this._secret(secrets, auth.bearer)}`;
    } else if (auth.basic != null) {
      const username = this._secret(secrets, auth.basic.username);
      if (username.includes(":")) {
        throw new Error("configured connector basic username is invalid");
      }
      const value = `${username}:${this._secret(secrets, auth.basic.password)}`;
      headers["Authorization"] =
        `Basic ${Buffer.from(value, "ascii").toString("base64")}`;
    } else {
      for (const credential of auth.api_keys ?? []) {
        const value = credential.prefix + this._secret(secrets, credential.env);
        if (credential.location === "header") {
          headers[credential.name] = value;
        } else if (credential.location === "query") {
          const pair: [string, string] = [credential.name, value];
          authQuery.push(pair);
          query.push(pair);
        } else if (credential.location === "cookie") {
          const pair = `${credential.name}=${quote(value, false)}`;
          headers["Cookie"] = [headers["Cookie"], pair]
            .filter((item): item is string => Boolean(item))
            .join("; ");
        } else {
          const encoded = quote(validateSegment(value), false);
          hasPathAuth = true;
          baseUrl = baseUrl.split(`{${credential.name}}`).join(encoded);
        }
      }
    }
    Object.assign(headers, auth.headers ?? {});
    return { baseUrl, authQuery, hasPathAuth };
  }

  private _requestBody(
    tool: Operation,
    arguments_: Record<string, unknown>,
  ): { bytes: Uint8Array | undefined; contentType?: string } {
    if (tool.params.wrapped && "body" in arguments_) {
      const body = arguments_["body"];
      if (tool.body_encoding === "form") {
        if (!isRecord(body)) {
          throw new Error("form request body must be an object");
        }
        return {
          bytes: Buffer.from(
            this._formFields(body, tool.params.form_style)
              .map(
                ([key, value]) => `${quote(key, true)}=${quote(value, true)}`,
              )
              .join("&"),
            "utf8",
          ),
          contentType: "application/x-www-form-urlencoded",
        };
      }
      if (tool.body_encoding === "multipart") {
        if (!isRecord(body)) {
          throw new Error("multipart request body must be an object");
        }
        return this._multipartBody(body);
      }
      if (tool.body_encoding === "text") {
        if (typeof body !== "string") {
          throw new Error("text request body must be a string");
        }
        return {
          bytes: Buffer.from(body, "utf8"),
          contentType: "text/plain",
        };
      }
      if (tool.body_encoding === "binary") {
        const value = this._scalar(body);
        if (!/^[A-Za-z0-9+/]*={0,2}$/.test(value) || value.length % 4 !== 0) {
          throw new Error("binary request body must be valid base64");
        }
        return {
          bytes: Buffer.from(value, "base64"),
          contentType: "application/octet-stream",
        };
      }
      return {
        bytes: Buffer.from(JSON.stringify(body), "utf8"),
        contentType: tool.content_type ?? "application/json",
      };
    }

    if (tool.params.body.length > 0) {
      const body = Object.fromEntries(
        tool.params.body
          .filter((key) => key in arguments_)
          .map((key) => [key, arguments_[key]]),
      );
      if (Object.keys(body).length > 0) {
        return {
          bytes: Buffer.from(JSON.stringify(body), "utf8"),
          contentType: tool.content_type ?? "application/json",
        };
      }
    }
    return { bytes: undefined };
  }

  private _formFields(
    body: Record<string, unknown>,
    styles: Record<string, string>,
  ): Array<[string, string]> {
    const fields: Array<[string, string]> = [];
    for (const [key, value] of Object.entries(body)) {
      if (value === null || value === undefined) continue;
      const style = styles[key];
      if (style === "brackets" || isRecord(value)) {
        fields.push(...this._bracketPairs(key, value));
      } else if (Array.isArray(value)) {
        const rendered = value
          .filter((item) => item !== null && item !== undefined)
          .map((item) => this._scalar(item));
        if (style === "comma") fields.push([key, rendered.join(",")]);
        else for (const item of rendered) fields.push([key, item]);
      } else {
        fields.push([key, this._scalar(value)]);
      }
    }
    return fields;
  }

  private _multipartBody(body: Record<string, unknown>): {
    bytes: Uint8Array;
    contentType: string;
  } {
    const boundary = `----agent-engine-${randomUUID()}`;
    const chunks: Buffer[] = [];
    for (const [key, value] of Object.entries(body)) {
      if (value === null || value === undefined) continue;
      const structured = isRecord(value) || Array.isArray(value);
      const rendered = structured ? JSON.stringify(value) : this._scalar(value);
      chunks.push(
        Buffer.from(
          `--${boundary}\r\n` +
            `Content-Disposition: form-data; name="${escapeMultipartName(key)}"\r\n` +
            `Content-Type: ${structured ? "application/json" : "text/plain"}\r\n\r\n`,
          "utf8",
        ),
        Buffer.from(rendered, "utf8"),
        Buffer.from("\r\n", "utf8"),
      );
    }
    chunks.push(Buffer.from(`--${boundary}--\r\n`, "utf8"));
    return {
      bytes: Buffer.concat(chunks),
      contentType: `multipart/form-data; boundary=${boundary}`,
    };
  }

  private _bracketPairs(
    prefix: string,
    value: unknown,
  ): Array<[string, string]> {
    if (isRecord(value)) {
      const pairs: Array<[string, string]> = [];
      for (const [key, item] of Object.entries(value)) {
        pairs.push(
          ...this._bracketPairs(`${prefix}[${this._scalar(key)}]`, item),
        );
      }
      return pairs;
    }
    if (Array.isArray(value)) {
      const pairs: Array<[string, string]> = [];
      for (const item of value)
        pairs.push(...this._bracketPairs(`${prefix}[]`, item));
      return pairs;
    }
    if (value === null || value === undefined) return [];
    return [[prefix, this._scalar(value)]];
  }

  private _request(
    method: string,
    startingUrl: string,
    startingHeaders: Record<string, string>,
    startingBody: Uint8Array | undefined,
    deadline: number,
    authQuery: Array<[string, string]>,
    hasPathAuth: boolean,
  ): Promise<{
    status: number;
    headers: { get(name: string): string | null };
    text: string;
  }> {
    const starting = new URL(startingUrl);
    const startingOrigin = originOf(starting);
    const redirectLimit = MAX_REDIRECTS + 1;
    return (async () => {
      let url = starting;
      let currentMethod = method;
      let currentHeaders = { ...startingHeaders };
      let currentBody = startingBody;
      for (let redirect = 0; redirect < redirectLimit; redirect += 1) {
        const remaining = this._remainingSeconds(deadline);
        const signal = AbortSignal.timeout(
          Math.max(1, Math.ceil(remaining * 1000)),
        );
        let response: ConnectorTransportResponse;
        try {
          response = await this.transport({
            method: currentMethod,
            url: url.toString(),
            headers: currentHeaders,
            body: currentBody,
            signal,
          });
        } catch (error) {
          if (
            isTimeoutFailure(error) ||
            signal.aborted ||
            this.now() >= deadline
          ) {
            throw new ConnectorTimeoutError();
          }
          throw new ConnectorTransportError("connector transport failed");
        }

        if (REDIRECT_STATUSES.has(response.status)) {
          const location = response.headers.get("location");
          if (!location || redirect === MAX_REDIRECTS) {
            await discardBody(response.body);
            throw this._httpError(response.status);
          }
          let nextUrl: URL;
          try {
            nextUrl = new URL(location, url);
            validateUrl(nextUrl.toString());
          } catch {
            await discardBody(response.body);
            throw this._httpError(response.status);
          }
          if (originOf(nextUrl) !== startingOrigin || hasPathAuth) {
            await discardBody(response.body);
            throw this._httpError(response.status);
          }
          if (authQuery.length > 0) {
            const authNames = new Set(authQuery.map(([name]) => name));
            const params = new URLSearchParams(nextUrl.search);
            for (const name of authNames) params.delete(name);
            for (const [name, value] of authQuery) params.append(name, value);
            nextUrl.search = params.toString();
          }
          await discardBody(response.body);
          if (
            (response.status === 303 && currentMethod !== "HEAD") ||
            ((response.status === 301 || response.status === 302) &&
              currentMethod === "POST")
          ) {
            currentMethod = "GET";
            currentBody = undefined;
            currentHeaders = Object.fromEntries(
              Object.entries(currentHeaders).filter(
                ([name]) => name.toLowerCase() !== "content-type",
              ),
            );
          }
          url = nextUrl;
          continue;
        }

        try {
          const text = await readBoundedBody(
            response.body,
            deadline,
            this.now,
            signal,
          );
          return {
            status: response.status,
            headers: response.headers,
            text,
          };
        } catch (error) {
          await discardBody(response.body);
          if (
            isTimeoutFailure(error) ||
            signal.aborted ||
            this.now() >= deadline
          ) {
            throw new ConnectorTimeoutError();
          }
          throw new ConnectorTransportError(
            "connector response decoding failed",
          );
        }
      }
      throw new Error("redirect loop exhausted without a result");
    })();
  }

  private _scalar(value: unknown): string {
    if (
      typeof value !== "string" &&
      typeof value !== "number" &&
      typeof value !== "boolean"
    ) {
      throw new Error("non-body arguments must be scalar values");
    }
    const result =
      typeof value === "boolean" ? String(value).toLowerCase() : String(value);
    for (const character of result) {
      const code = character.codePointAt(0) ?? 0;
      if (code < 32 || (code >= 127 && code <= 159)) {
        throw new Error("parameter contains control characters");
      }
    }
    return result;
  }

  private _paramValue(
    arguments_: Record<string, unknown>,
    key: string,
  ): string | string[] {
    const value = arguments_[key];
    if (Array.isArray(value)) return value.map((item) => this._scalar(item));
    return this._scalar(value);
  }

  private _secret(env: Readonly<Record<string, string>>, name: string): string {
    const value = env[name];
    if (!value || !isPrintableAscii(value)) {
      throw new Error("configured connector secret is missing or invalid");
    }
    return value;
  }

  private _serializeQuery(pairs: Array<[string, string]>): string {
    return pairs
      .map(([key, value]) => `${quote(key, true)}=${quote(value, true)}`)
      .join("&");
  }

  private _remainingSeconds(deadline: number): number {
    const remaining = (deadline - this.now()) / 1000;
    if (remaining <= 0) throw new ConnectorTimeoutError();
    return remaining;
  }

  private _httpError(status: number, body?: string): ConnectorHTTPError {
    let data: unknown;
    if (
      body !== undefined &&
      Buffer.byteLength(body, "utf8") <= MAX_ERROR_ENVELOPE_BYTES
    ) {
      try {
        const parsed: unknown = JSON.parse(body);
        if (typeof parsed === "object" && parsed !== null) data = parsed;
        else if (typeof parsed === "string") data = parsed;
      } catch {
        data = body;
      }
    }
    return new ConnectorHTTPError(status, data);
  }
}

function isPrintableAscii(value: string): boolean {
  for (const character of value) {
    const code = character.codePointAt(0) ?? 0;
    if (code < 32 || code >= 127) return false;
  }
  return true;
}

/** Delegated tokens are header values: whitespace is not a legal token byte. */
function isDelegatedToken(value: string): boolean {
  for (const character of value) {
    const code = character.codePointAt(0) ?? 0;
    if (code <= 32 || code >= 127) return false;
  }
  return true;
}

function escapeMultipartName(value: string): string {
  // Mirror HTTPX's HTML5 form encoding: quote and backslash are escaped, and
  // C0 controls (including CR/LF) are percent-encoded so a field name cannot
  // become additional multipart headers or framing.
  let escaped = "";
  for (const character of value) {
    if (character === '"') {
      escaped += "%22";
    } else if (character === "\\") {
      escaped += "\\\\";
    } else {
      const code = character.codePointAt(0) ?? 0;
      escaped +=
        code <= 0x1f
          ? `%${code.toString(16).toUpperCase().padStart(2, "0")}`
          : character;
    }
  }
  return escaped;
}
