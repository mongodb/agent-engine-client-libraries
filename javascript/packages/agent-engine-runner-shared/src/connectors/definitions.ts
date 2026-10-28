/**
 * Stable compiled connector contract consumed at runtime.
 *
 * The CLI's OpenAPI generator writes these definitions to `tool_defs.yaml`;
 * the AER and Tool Pod read them back. OpenAPI concepts do not cross this
 * boundary: this module validates only tool arguments, HTTP placement,
 * request-body encoding, and materialized credential bindings.
 *
 * Mirrors `agent_engine_runner_shared/connectors/definitions.py`.
 */

import { readFileSync } from "node:fs";
import { isIP } from "node:net";

import { Ajv2020 } from "ajv/dist/2020.js";
import * as yaml from "js-yaml";
import { z } from "zod";

import { isPlatformEnvVar } from "../utils.js";

const HEADER_NAME_RE = /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/;
const RESERVED_HEADERS = new Set([
  "authorization",
  "proxy-authorization",
  "host",
  "cookie",
  "content-length",
  "transfer-encoding",
  "connection",
  "content-type",
]);
const TOOL_ARGUMENT_SCHEMA_KEYS = new Set([
  "type",
  "properties",
  "required",
  "$defs",
]);
const SCALAR_PARAM_TYPES = new Set(["string", "number", "integer", "boolean"]);
const CONNECTOR_NAME_RE = /^[A-Za-z0-9_-]{1,64}$/;
const SECRET_ENV_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const OAUTH_SCOPE_RE = /^[!\x23-\x5B\x5D-\x7E]+$/;
const AJV = new Ajv2020({
  strict: false,
  validateFormats: false,
  allErrors: false,
  // Python's Draft 2020-12 checker accepts identity escapes such as `\_` in
  // generated patterns; Ajv's Unicode-mode RegExp rejects them.
  unicodeRegExp: false,
});

// Python uses two loaders: compiled catalogs go through StrictSafeLoader
// (PyYAML's YAML 1.1 scalars without bool/timestamp coercion), while
// authoring `tool.yaml` files go through yaml.safe_load. YAML 1.2 core is not
// equivalent for either: it parses values such as `1e3` and `08` as numbers,
// where PyYAML keeps them as strings. These resolvers mirror PyYAML.
const PYYAML_INT_RE =
  /^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$/;
const PYYAML_FLOAT_RE =
  /^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9_]+(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$/;
const PYYAML_SAFE_BOOL_RE =
  /^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$/;
const PYYAML_SAFE_TRUE_RE = /^(?:yes|Yes|YES|true|True|TRUE|on|On|ON)$/;
// @types/js-yaml does not expose the tags a Schema carries. The runtime
// objects do, and these are the tags this module reuses unchanged.
const DEFAULT_YAML_SCHEMA = yaml.DEFAULT_SCHEMA as unknown as {
  implicit: Array<yaml.Type & { tag: string }>;
  explicit: yaml.Type[];
};

function parsePyYamlInt(data: string): number {
  const clean = data.replace(/_/g, "");
  const sign = clean.startsWith("-") ? -1 : 1;
  const body = clean.replace(/^[-+]/, "");
  if (body === "0") return 0;
  if (body.startsWith("0b")) return sign * Number.parseInt(body.slice(2), 2);
  if (body.startsWith("0x")) return sign * Number.parseInt(body.slice(2), 16);
  if (body.startsWith("0") && body.length > 1)
    return sign * Number.parseInt(body, 8);
  if (body.includes(":")) {
    return (
      sign *
      body
        .split(":")
        .reverse()
        .reduce((total, part, index) => total + Number(part) * 60 ** index, 0)
    );
  }
  return sign * Number.parseInt(body, 10);
}

function parsePyYamlFloat(data: string): number {
  const clean = data.replace(/_/g, "").toLowerCase();
  const sign = clean.startsWith("-") ? -1 : 1;
  const body = clean.replace(/^[-+]/, "");
  if (body === ".inf") return sign * Number.POSITIVE_INFINITY;
  if (body === ".nan") return Number.NaN;
  if (body.includes(":")) {
    return (
      sign *
      body
        .split(":")
        .reverse()
        .reduce((total, part, index) => total + Number(part) * 60 ** index, 0)
    );
  }
  return sign * Number(body);
}

function defaultImplicitType(tag: string): yaml.Type {
  const type = DEFAULT_YAML_SCHEMA.implicit.find(
    (candidate) => candidate.tag === tag,
  );
  if (type === undefined) throw new Error(`js-yaml is missing ${tag}`);
  return type;
}

const pyYamlNullType = new yaml.Type("tag:yaml.org,2002:null", {
  kind: "scalar",
  resolve: (data: string) => data === "" || /^(?:~|null|Null|NULL)$/.test(data),
  construct: () => null,
});
const pyYamlStrictBoolType = new yaml.Type("tag:yaml.org,2002:bool", {
  kind: "scalar",
  resolve: (data: string) =>
    /^(?:true|True|TRUE|false|False|FALSE)$/.test(data),
  construct: (data: string) => /^(?:true|True|TRUE)$/.test(data),
});
const pyYamlSafeBoolType = new yaml.Type("tag:yaml.org,2002:bool", {
  kind: "scalar",
  resolve: (data: string) => PYYAML_SAFE_BOOL_RE.test(data),
  construct: (data: string) => PYYAML_SAFE_TRUE_RE.test(data),
});
const pyYamlIntType = new yaml.Type("tag:yaml.org,2002:int", {
  kind: "scalar",
  resolve: (data: string) => PYYAML_INT_RE.test(data),
  construct: (data: string) => parsePyYamlInt(data),
});
const pyYamlFloatType = new yaml.Type("tag:yaml.org,2002:float", {
  kind: "scalar",
  resolve: (data: string) => PYYAML_FLOAT_RE.test(data),
  construct: (data: string) => parsePyYamlFloat(data),
});
// PyYAML reads the bare `=` scalar as a string, not the YAML 1.2 tag value.
const pyYamlValueType = new yaml.Type("tag:yaml.org,2002:value", {
  kind: "scalar",
  resolve: (data: string) => data === "=",
  construct: () => "=",
});
const STRICT_YAML_SCHEMA = new yaml.Schema({
  implicit: [
    pyYamlNullType,
    pyYamlStrictBoolType,
    pyYamlIntType,
    pyYamlFloatType,
    pyYamlValueType,
  ],
  explicit: DEFAULT_YAML_SCHEMA.explicit,
});
const AUTHORING_YAML_SCHEMA = new yaml.Schema({
  implicit: [
    pyYamlNullType,
    pyYamlSafeBoolType,
    pyYamlIntType,
    pyYamlFloatType,
    pyYamlValueType,
    defaultImplicitType("tag:yaml.org,2002:timestamp"),
    defaultImplicitType("tag:yaml.org,2002:merge"),
  ],
  explicit: DEFAULT_YAML_SCHEMA.explicit,
});

/** Parse a compiled catalog with Python StrictSafeLoader semantics. */
export function loadConnectorYaml(text: string): unknown {
  return yaml.load(text, { schema: STRICT_YAML_SCHEMA });
}

/** Parse authoring `tool.yaml` with Python `yaml.safe_load` semantics. */
export function loadAuthoringYaml(text: string): unknown {
  return yaml.load(text, { schema: AUTHORING_YAML_SCHEMA });
}

const CONNECTOR_NAME_PATTERN = "^[A-Za-z0-9_-]{1,64}$";

export class ConnectorDefinitionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConnectorDefinitionError";
  }
}

/** A plain JSON object, as opposed to an array or `null`. */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function hasControlChars(value: string): boolean {
  for (const character of value) {
    if ((character.codePointAt(0) ?? 0) <= 32) return true;
  }
  return false;
}

function hasControlOrNonAscii(value: string): boolean {
  for (const character of value) {
    const code = character.codePointAt(0) ?? 0;
    if (code < 32 || code >= 127) return true;
  }
  return false;
}

function isLoopbackHostname(hostname: string): boolean {
  const host = hostname.replace(/^\[|\]$/g, "").toLowerCase();
  if (host === "localhost" || host === "::1") return true;
  const family = isIP(host);
  if (family === 4) {
    const octets = host.split(".").map(Number);
    return octets[0] === 127;
  }
  return false;
}

/** Require HTTPS, or HTTP only for loopback, with no userinfo or fragment. */
export function validateUrl(value: string): string {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new ConnectorDefinitionError(
      "connector URL requires HTTPS (HTTP allowed only for loopback), no userinfo or fragment",
    );
  }
  if (
    !parsed.hostname ||
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.hash !== "" ||
    value.includes("\\") ||
    hasControlChars(value) ||
    !(
      parsed.protocol === "https:" ||
      (parsed.protocol === "http:" && isLoopbackHostname(parsed.hostname))
    )
  ) {
    throw new ConnectorDefinitionError(
      "connector URL requires HTTPS (HTTP allowed only for loopback), no userinfo or fragment",
    );
  }
  return value;
}

/** Reject traversal and separators, including percent-encoded spellings. */
export function validateSegment(value: string): string {
  if (value.toLowerCase().includes("%25")) {
    throw new ConnectorDefinitionError(
      "path parameter must be a single non-traversing segment",
    );
  }
  // Python's unquote decodes every valid escape and leaves malformed ones
  // literal. decodeURIComponent rejects the whole string on one malformed
  // escape, which would hide separators in mixed input.
  const decoded = value.replace(/(?:%[0-9A-Fa-f]{2})+/g, (match) => {
    const bytes = new Uint8Array(match.length / 3);
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Number.parseInt(
        match.slice(index * 3 + 1, index * 3 + 3),
        16,
      );
    }
    return new TextDecoder("utf-8").decode(bytes);
  });
  if (decoded === "." || decoded === ".." || /[/\\]/.test(decoded)) {
    throw new ConnectorDefinitionError(
      "path parameter must be a single non-traversing segment",
    );
  }
  return value;
}

/** Connector names key connector identity in runtime registration. */
export function validateConnectorName(value: string): string {
  if (!CONNECTOR_NAME_RE.test(value)) {
    throw new ConnectorDefinitionError(
      `connector name must match ${CONNECTOR_NAME_PATTERN}`,
    );
  }
  return value;
}

function isSecretEnvReference(value: string | null | undefined): boolean {
  return Boolean(
    value && SECRET_ENV_RE.test(value) && !isPlatformEnvVar(value),
  );
}

function validateStaticHeaders(
  headers: Record<string, string> | null | undefined,
  credentialHeaders: Set<string>,
): void {
  for (const [name, value] of Object.entries(headers ?? {})) {
    if (
      !HEADER_NAME_RE.test(name) ||
      RESERVED_HEADERS.has(name.toLowerCase())
    ) {
      throw new ConnectorDefinitionError(
        "static auth headers cannot control transport headers",
      );
    }
    if (!value || hasControlOrNonAscii(value)) {
      throw new ConnectorDefinitionError(
        "static auth header values must contain printable ASCII",
      );
    }
  }
  const headerNames = new Set(
    Object.keys(headers ?? {}).map((name) => name.toLowerCase()),
  );
  for (const credentialHeader of credentialHeaders) {
    if (headerNames.has(credentialHeader)) {
      throw new ConnectorDefinitionError(
        "static auth headers collide with credential headers",
      );
    }
  }
}

export const CredentialSchema = z
  .object({
    env: z.string(),
    location: z.enum(["header", "query", "path", "cookie"]),
    name: z.string().min(1),
    prefix: z.string().default(""),
  })
  .strict()
  .superRefine((value, ctx) => {
    if (!isSecretEnvReference(value.env)) {
      ctx.addIssue({
        code: "custom",
        message: "credential requires a tenant-owned environment reference",
      });
    }
    if (hasControlOrNonAscii(value.prefix)) {
      ctx.addIssue({
        code: "custom",
        message: "credential prefix must contain printable ASCII",
      });
    }
    if (
      (value.location === "header" || value.location === "cookie") &&
      !HEADER_NAME_RE.test(value.name)
    ) {
      ctx.addIssue({
        code: "custom",
        message: "credential header and cookie names must be HTTP tokens",
      });
    }
    if (
      value.location === "header" &&
      RESERVED_HEADERS.has(value.name.toLowerCase()) &&
      value.name.toLowerCase() !== "authorization"
    ) {
      ctx.addIssue({
        code: "custom",
        message: "credential cannot use a transport-controlled header",
      });
    }
    if (
      (value.location === "query" || value.location === "path") &&
      [...value.name].some((character) => {
        const code = character.codePointAt(0) ?? 0;
        return code <= 32 || "#&?/{}`\\".includes(character);
      })
    ) {
      ctx.addIssue({
        code: "custom",
        message: "credential query and path names contain unsafe characters",
      });
    }
  });
export type Credential = z.infer<typeof CredentialSchema>;

export type AuthType =
  | "none"
  | "bearer"
  | "api_key"
  | "basic"
  | "api_keys"
  | "oauth";

const ALLOWED_AUTH_FIELDS: Record<AuthType, ReadonlySet<string>> = {
  none: new Set(["type"]),
  bearer: new Set(["type", "env", "headers"]),
  api_key: new Set([
    "type",
    "env",
    "header",
    "location",
    "name",
    "prefix",
    "headers",
  ]),
  basic: new Set(["type", "username_env", "password_env", "headers"]),
  api_keys: new Set(["type", "credentials", "headers"]),
  oauth: new Set(["type", "provider_type", "scopes", "headers"]),
};

export const AuthSchema = z
  .object({
    type: z
      .enum(["none", "bearer", "api_key", "basic", "api_keys", "oauth"])
      .default("none"),
    provider_type: z.string().nullable().optional(),
    scopes: z.array(z.string()).default([]),
    env: z.string().nullable().optional(),
    header: z.string().nullable().optional(),
    location: z
      .enum(["header", "query", "path", "cookie"])
      .nullable()
      .optional(),
    name: z.string().nullable().optional(),
    prefix: z.string().default(""),
    username_env: z.string().nullable().optional(),
    password_env: z.string().nullable().optional(),
    credentials: z.array(CredentialSchema).min(2).nullable().optional(),
    headers: z.record(z.string(), z.string()).nullable().optional(),
  })
  .strict()
  .superRefine((value, ctx) => {
    const configured = new Set<string>(["type"]);
    for (const [name, fieldValue] of Object.entries(value)) {
      if (name === "type" || fieldValue === undefined || fieldValue === null)
        continue;
      if (typeof fieldValue === "string" && fieldValue === "") continue;
      if (Array.isArray(fieldValue) && fieldValue.length === 0) continue;
      if (isRecord(fieldValue) && Object.keys(fieldValue).length === 0)
        continue;
      configured.add(name);
    }
    const allowed = ALLOWED_AUTH_FIELDS[value.type];
    const unexpected = [...configured].filter((name) => !allowed.has(name));
    if (unexpected.length > 0) {
      ctx.addIssue({
        code: "custom",
        message: `${value.type} auth contains fields for another auth type`,
      });
    }

    let credentialHeaders = new Set<string>();
    if (value.type === "oauth") {
      if (
        !value.provider_type ||
        hasControlOrNonAscii(value.provider_type) ||
        /\s/.test(value.provider_type)
      ) {
        ctx.addIssue({
          code: "custom",
          message:
            "OAuth auth requires an explicit provider key without whitespace",
        });
      }
      if (
        new Set(value.scopes).size !== value.scopes.length ||
        value.scopes.some((scope) => !OAUTH_SCOPE_RE.test(scope))
      ) {
        ctx.addIssue({
          code: "custom",
          message: "OAuth scopes must be unique OAuth scope tokens",
        });
      }
      credentialHeaders = new Set(["authorization"]);
    } else if (value.type === "bearer") {
      if (!isSecretEnvReference(value.env)) {
        ctx.addIssue({
          code: "custom",
          message: "bearer auth requires one tenant secret env",
        });
      }
      credentialHeaders = new Set(["authorization"]);
    } else if (value.type === "api_key") {
      const location = value.location ?? "header";
      const name = value.name ?? value.header ?? "X-API-Key";
      if (
        value.header !== undefined &&
        value.header !== null &&
        (location !== "header" ||
          (value.name !== undefined && value.name !== null))
      ) {
        ctx.addIssue({
          code: "custom",
          message:
            "legacy auth.header cannot be combined with location or name",
        });
      }
      if (!isSecretEnvReference(value.env)) {
        ctx.addIssue({
          code: "custom",
          message: "API key auth requires a tenant-owned environment reference",
        });
      }
      if (value.env) {
        const credential = CredentialSchema.safeParse({
          env: value.env,
          location,
          name,
          prefix: value.prefix,
        });
        if (!credential.success) {
          for (const issue of credential.error.issues) {
            ctx.addIssue({ code: "custom", message: issue.message });
          }
        }
      }
      if (location === "header")
        credentialHeaders = new Set([name.toLowerCase()]);
    } else if (value.type === "basic") {
      if (
        !isSecretEnvReference(value.username_env) ||
        !isSecretEnvReference(value.password_env)
      ) {
        ctx.addIssue({
          code: "custom",
          message:
            "basic auth requires tenant-owned username and password envs",
        });
      }
      if (value.username_env === value.password_env) {
        ctx.addIssue({
          code: "custom",
          message:
            "basic auth username and password must use different secrets",
        });
      }
      credentialHeaders = new Set(["authorization"]);
    } else if (value.type === "api_keys") {
      if (!value.credentials || value.credentials.length === 0) {
        ctx.addIssue({
          code: "custom",
          message: "multi-key auth requires at least two credentials",
        });
      }
      const identities = (value.credentials ?? []).map(
        (credential) =>
          `${credential.location}:${credential.name.toLowerCase()}`,
      );
      if (identities.length !== new Set(identities).size) {
        ctx.addIssue({
          code: "custom",
          message: "multi-key auth has duplicate credential destinations",
        });
      }
      credentialHeaders = new Set(
        (value.credentials ?? [])
          .filter((credential) => credential.location === "header")
          .map((credential) => credential.name.toLowerCase()),
      );
    }

    try {
      validateStaticHeaders(value.headers, credentialHeaders);
    } catch (error) {
      ctx.addIssue({
        code: "custom",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  });
export type Auth = z.infer<typeof AuthSchema>;

/** Return the single generated auth header, when this auth shape has one. */
export function authHeaderName(auth: Auth | null | undefined): string | null {
  if (!auth) return null;
  if (auth.type === "bearer" || auth.type === "oauth") return "Authorization";
  if (auth.type === "api_key" && (auth.location ?? "header") === "header") {
    return auth.name ?? auth.header ?? "X-API-Key";
  }
  return null;
}

/** List auth-controlled headers so model parameters cannot collide with them. */
export function authHeaderNames(auth: Auth | null | undefined): Set<string> {
  if (!auth) return new Set();
  let names: Set<string>;
  if (
    auth.type === "bearer" ||
    auth.type === "basic" ||
    auth.type === "oauth"
  ) {
    names = new Set(["authorization"]);
  } else if (auth.type === "api_key") {
    const header = authHeaderName(auth);
    names = header ? new Set([header.toLowerCase()]) : new Set();
  } else if (auth.type === "api_keys") {
    names = new Set(
      (auth.credentials ?? [])
        .filter((credential) => credential.location === "header")
        .map((credential) => credential.name.toLowerCase()),
    );
  } else {
    names = new Set();
  }
  for (const header of Object.keys(auth.headers ?? {}))
    names.add(header.toLowerCase());
  return names;
}

/** Normalize single and compound API keys into the executor's common form. */
export function authCredentialBindings(
  auth: Auth | null | undefined,
): Credential[] {
  if (!auth) return [];
  if (auth.type === "api_key" && auth.env) {
    return [
      {
        env: auth.env,
        location: auth.location ?? "header",
        name: auth.name ?? auth.header ?? "X-API-Key",
        prefix: auth.prefix,
      },
    ];
  }
  return [...(auth.credentials ?? [])];
}

export const ParametersSchema = z
  .object({
    path: z.array(z.string()).default([]),
    query: z.array(z.string()).default([]),
    header: z.array(z.string()).default([]),
    cookie: z.array(z.string()).default([]),
    body: z.array(z.string()).default([]),
    wrapped: z.boolean().default(false),
    query_style: z.record(z.string(), z.string()).default({}),
    form_style: z.record(z.string(), z.string()).default({}),
  })
  .strict();
export type Parameters = z.infer<typeof ParametersSchema>;

function nonNullType(propertySchema: Record<string, unknown>): unknown {
  const kind = propertySchema["type"];
  if (Array.isArray(kind) && kind.includes("null")) {
    const members = kind.filter((member) => member !== "null");
    if (members.length === 1) return members[0];
  }
  return kind;
}

function unionBranches(schema: Record<string, unknown>): unknown[] | null {
  for (const keyword of ["anyOf", "oneOf"]) {
    const branches = schema[keyword];
    if (!Array.isArray(branches) || branches.length === 0) continue;
    const allowedKeys = new Set([keyword, "description", "title", "default"]);
    const hasUnexpectedKey = Object.keys(schema).some(
      (key) => !allowedKeys.has(key) && !key.startsWith("x-"),
    );
    if (hasUnexpectedKey) return null;
    return branches;
  }
  return null;
}

function paramBranchOk(
  branch: unknown,
  options: { allowObject: boolean; allowArray: boolean },
): boolean {
  if (!isRecord(branch)) return false;
  const kind = nonNullType(branch);
  if (typeof kind === "string" && SCALAR_PARAM_TYPES.has(kind)) return true;
  if (kind === "array") {
    const items = branch["items"];
    return (
      options.allowArray &&
      isRecord(items) &&
      typeof items["type"] === "string" &&
      SCALAR_PARAM_TYPES.has(items["type"])
    );
  }
  return options.allowObject && kind === "object";
}

function paramSchemaOk(
  schema: unknown,
  options: { allowObject: boolean; allowArray: boolean },
): boolean {
  if (!isRecord(schema)) return false;
  if (paramBranchOk(schema, options)) return true;
  const branches = unionBranches(schema);
  if (branches === null) return false;
  return branches.every((branch) => paramBranchOk(branch, options));
}

function checkLocalSchemaReferences(
  schema: unknown,
  localDefs?: Set<string>,
): void {
  if (Array.isArray(schema)) {
    for (const value of schema) checkLocalSchemaReferences(value, localDefs);
    return;
  }
  if (!isRecord(schema)) return;
  const definitions =
    localDefs ??
    new Set(Object.keys(isRecord(schema["$defs"]) ? schema["$defs"] : {}));
  const reference = schema["$ref"];
  if (reference !== undefined) {
    if (
      typeof reference !== "string" ||
      !reference.startsWith("#/$defs/") ||
      !definitions.has(reference.slice("#/$defs/".length))
    ) {
      throw new ConnectorDefinitionError(
        "compiled schemas may reference only local definitions",
      );
    }
  }
  if ("$dynamicRef" in schema || "$recursiveRef" in schema) {
    throw new ConnectorDefinitionError(
      "compiled schemas may reference only local definitions",
    );
  }
  for (const value of Object.values(schema))
    checkLocalSchemaReferences(value, definitions);
}

function assertJsonValue(value: unknown, seen: Set<object> = new Set()): void {
  if (value === null) return;
  if (typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") {
    if (!Number.isFinite(value))
      throw new ConnectorDefinitionError(
        "compiled input schema must contain JSON values",
      );
    return;
  }
  if (typeof value !== "object") {
    throw new ConnectorDefinitionError(
      "compiled input schema must contain JSON values",
    );
  }
  if (seen.has(value))
    throw new ConnectorDefinitionError(
      "compiled input schema must contain JSON values",
    );
  seen.add(value);
  if (Array.isArray(value)) {
    for (const item of value) assertJsonValue(item, seen);
  } else {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new ConnectorDefinitionError(
        "compiled input schema must contain JSON values",
      );
    }
    for (const item of Object.values(value)) assertJsonValue(item, seen);
  }
  seen.delete(value);
}

function compileInputSchema(schema: Record<string, unknown>): void {
  try {
    AJV.compile(schema);
  } catch {
    throw new ConnectorDefinitionError("invalid compiled input schema");
  }
}

export const OperationSchema = z
  .object({
    name: z.string().regex(/^[A-Za-z0-9_-]{1,64}$/),
    description: z.string().default(""),
    method: z.enum([
      "GET",
      "HEAD",
      "OPTIONS",
      "POST",
      "PUT",
      "PATCH",
      "DELETE",
    ]),
    path: z.string(),
    params: ParametersSchema,
    inputSchema: z.record(z.string(), z.unknown()),
    body_encoding: z
      .enum(["json", "form", "multipart", "text", "binary"])
      .default("json"),
    content_type: z.string().nullable().optional(),
  })
  .strict()
  .superRefine((value, ctx) => {
    const issue = (message: string): void => {
      ctx.addIssue({ code: "custom", message });
    };
    if (
      value.content_type !== undefined &&
      value.content_type !== null &&
      (value.body_encoding !== "json" ||
        !value.content_type.startsWith("application/") ||
        !value.content_type.endsWith("+json") ||
        [...value.content_type].some(
          (character) => (character.codePointAt(0) ?? 0) <= 32,
        ))
    ) {
      issue("content type must be a vendor application +json media type");
    }
    if (
      !value.path.startsWith("/") ||
      value.path.startsWith("//") ||
      /[?#\\]/.test(value.path) ||
      hasControlChars(value.path)
    ) {
      issue(
        "operation path must be an absolute API path without query or fragment",
      );
    }
    for (const segment of value.path.split("/")) {
      try {
        validateSegment(segment);
      } catch (error) {
        issue(error instanceof Error ? error.message : String(error));
      }
    }
    const placeholders = [...value.path.matchAll(/\{([^{}]+)\}/g)].map(
      (match) => match[1],
    );
    const withoutPlaceholders = value.path.replace(/\{[^{}]+\}/g, "");
    if (
      new Set(placeholders).size !== new Set(value.params.path).size ||
      !placeholders.every((placeholder) =>
        value.params.path.includes(placeholder),
      ) ||
      /[{}]/.test(withoutPlaceholders)
    ) {
      issue("path placeholders must match path parameters");
    }
    const groups = [
      value.params.path,
      value.params.query,
      value.params.header,
      value.params.cookie,
      value.params.body,
      value.params.wrapped ? ["body"] : [],
    ];
    const names = groups.flat();
    if (
      names.length !== new Set(names).size ||
      (value.params.wrapped && value.params.body.length > 0)
    ) {
      issue("each argument must have exactly one placement");
    }
    if (
      Object.keys(value.params.query_style).some(
        (name) => !value.params.query.includes(name),
      )
    ) {
      issue("query serialization styles must name query parameters");
    }
    if (
      Object.values(value.params.query_style).some(
        (style) => !["comma", "brackets", "json"].includes(style),
      )
    ) {
      issue("unknown query serialization style");
    }
    if (
      Object.keys(value.params.form_style).length > 0 &&
      value.body_encoding !== "form"
    ) {
      issue("form serialization styles require a form body");
    }
    if (
      Object.values(value.params.form_style).some(
        (style) => !["comma", "brackets"].includes(style),
      )
    ) {
      issue("unknown form serialization style");
    }
    for (const name of [...value.params.header, ...value.params.cookie]) {
      if (!HEADER_NAME_RE.test(name))
        issue("invalid header or cookie parameter name");
    }
    const headerNames = value.params.header.map((name) => name.toLowerCase());
    if (
      new Set(headerNames).size !== headerNames.length ||
      headerNames.some((name) => RESERVED_HEADERS.has(name))
    ) {
      issue("header parameters cannot override auth or transport headers");
    }
    try {
      assertJsonValue(value.inputSchema);
      checkLocalSchemaReferences(value.inputSchema);
      compileInputSchema(value.inputSchema);
    } catch (error) {
      issue(error instanceof Error ? error.message : String(error));
    }
    if (
      Object.keys(value.inputSchema).some(
        (key) => !TOOL_ARGUMENT_SCHEMA_KEYS.has(key),
      )
    ) {
      issue("tool argument envelope contains unsupported keywords");
    }
    const properties = isRecord(value.inputSchema["properties"])
      ? value.inputSchema["properties"]
      : {};
    const required = Array.isArray(value.inputSchema["required"])
      ? value.inputSchema["required"].filter(
          (item): item is string => typeof item === "string",
        )
      : [];
    if (value.body_encoding === "form") {
      const bodySchema = properties["body"];
      const bodyProperties =
        isRecord(bodySchema) && isRecord(bodySchema["properties"])
          ? bodySchema["properties"]
          : null;
      if (
        !value.params.wrapped ||
        value.params.body.length > 0 ||
        bodyProperties === null
      ) {
        issue("form bodies must be wrapped objects");
      }
      if (bodyProperties) {
        for (const name of Object.keys(value.params.form_style)) {
          if (!(name in bodyProperties))
            issue("form serialization styles must name body properties");
        }
      }
    }
    if (
      value.inputSchema["type"] !== "object" ||
      Object.keys(properties).length !== new Set(names).size ||
      !Object.keys(properties).every((name) => names.includes(name))
    ) {
      issue("input schema properties must exactly match argument placement");
    }
    if (!value.params.path.every((name) => required.includes(name))) {
      issue("path parameters must be required");
    }
    if (
      !required.every((name) =>
        Object.prototype.hasOwnProperty.call(properties, name),
      )
    ) {
      issue("required arguments must have a placement");
    }
    for (const [placement, placementNames] of [
      ["path", value.params.path],
      ["query", value.params.query],
      ["header", value.params.header],
      ["cookie", value.params.cookie],
    ] as const) {
      for (const name of placementNames) {
        if (
          !paramSchemaOk(properties[name], {
            allowObject: placement === "query",
            allowArray: placement !== "cookie",
          })
        ) {
          issue(
            "non-body parameters support scalar, array, object, or union serialization only",
          );
        }
      }
    }
  });
export type Operation = z.infer<typeof OperationSchema>;

export const SourceSchema = z
  .object({
    name: z.string().min(1),
    base_url: z.string(),
    openapi: z.string().nullable().optional(),
    spec_sha256: z.string().nullable().optional(),
    generator: z.string().nullable().optional(),
  })
  .strict()
  .superRefine((value, ctx) => {
    try {
      validateUrl(value.base_url);
    } catch (error) {
      ctx.addIssue({
        code: "custom",
        message: error instanceof Error ? error.message : String(error),
      });
      return;
    }
    const parsed = new URL(value.base_url);
    if (parsed.search) {
      ctx.addIssue({
        code: "custom",
        message: "base URL cannot contain a query",
      });
    }
    const separator = value.base_url.indexOf("://");
    const afterScheme = value.base_url.slice(separator + 3);
    const pathStart = afterScheme.indexOf("/");
    const authority =
      pathStart >= 0 ? afterScheme.slice(0, pathStart) : afterScheme;
    if (/[{}]/.test(authority)) {
      ctx.addIssue({
        code: "custom",
        message: "base URL placeholders are allowed only in the URL path",
      });
      return;
    }
    const rawPath = pathStart >= 0 ? afterScheme.slice(pathStart) : "";
    const allPlaceholders = new Set(
      [...value.base_url.matchAll(/\{([^{}]+)\}/g)].map((m) => m[1]),
    );
    const pathPlaceholders = new Set(
      [...rawPath.matchAll(/\{([^{}]+)\}/g)].map((m) => m[1]),
    );
    if (
      allPlaceholders.size !== pathPlaceholders.size ||
      [...allPlaceholders].some(
        (placeholder) => !pathPlaceholders.has(placeholder),
      )
    ) {
      ctx.addIssue({
        code: "custom",
        message: "base URL placeholders are allowed only in the URL path",
      });
    }
  });
export type Source = z.infer<typeof SourceSchema>;

export const ToolDefinitionsSchema = z
  .object({
    source: SourceSchema,
    auth: AuthSchema.nullable().optional(),
    externalize: z.record(z.string(), z.unknown()).nullable().optional(),
    tools: z.array(OperationSchema).min(1),
  })
  .strict()
  .superRefine((value, ctx) => {
    const names = value.tools.map((tool) => tool.name);
    if (names.length !== new Set(names).size) {
      ctx.addIssue({ code: "custom", message: "duplicate tool names" });
    }
    const authHeaders = authHeaderNames(value.auth);
    for (const tool of value.tools) {
      for (const header of tool.params.header) {
        if (authHeaders.has(header.toLowerCase())) {
          ctx.addIssue({
            code: "custom",
            message: "auth header cannot be a model argument",
          });
        }
      }
    }
    const credentials = authCredentialBindings(value.auth);
    for (const credential of credentials) {
      if (
        credential.location === "query" &&
        value.tools.some((tool) => tool.params.query.includes(credential.name))
      ) {
        ctx.addIssue({
          code: "custom",
          message: "auth query parameter cannot be a model argument",
        });
      }
      if (
        credential.location === "cookie" &&
        value.tools.some((tool) => tool.params.cookie.includes(credential.name))
      ) {
        ctx.addIssue({
          code: "custom",
          message: "auth cookie cannot be a model argument",
        });
      }
    }
    const pathCredentials = new Set(
      credentials
        .filter((credential) => credential.location === "path")
        .map((credential) => credential.name),
    );
    const basePathPlaceholders = new Set(
      [...value.source.base_url.matchAll(/\{([^{}]+)\}/g)].map(
        (match) => match[1],
      ),
    );
    if (
      pathCredentials.size > 0 &&
      (pathCredentials.size !== basePathPlaceholders.size ||
        [...pathCredentials].some((name) => !basePathPlaceholders.has(name)))
    ) {
      ctx.addIssue({
        code: "custom",
        message:
          "base URL placeholders must exactly match path credential names",
      });
    }
  });
export type ToolDefinitions = z.infer<typeof ToolDefinitionsSchema>;

/** Format Zod issues into one bounded, path-qualified diagnostic. */
export function formatZodIssues(error: z.ZodError): string {
  return error.issues
    .map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`)
    .join("; ");
}

/** Return detached tool descriptions without credentials or transport metadata. */
export function toolSchemas(definitions: ToolDefinitions): Array<{
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}> {
  return definitions.tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    inputSchema: structuredClone(tool.inputSchema),
  }));
}

/** Read and validate a local compiled artifact; never load an OpenAPI source. */
export function loadToolDefs(path: string): ToolDefinitions {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    throw new ConnectorDefinitionError(
      `cannot read compiled connector catalog: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  let document: unknown;
  try {
    document = loadConnectorYaml(text);
  } catch (error) {
    throw new ConnectorDefinitionError(
      `invalid compiled connector catalog YAML: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const parsed = ToolDefinitionsSchema.safeParse(document);
  if (!parsed.success) {
    throw new ConnectorDefinitionError(
      `invalid compiled connector catalog: ${formatZodIssues(parsed.error)}`,
    );
  }
  return parsed.data;
}
