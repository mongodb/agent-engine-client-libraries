/** Exceptions raised by the memory seam. */

/** Raised when a required identity dimension cannot be resolved. */
export class MemoryIdentityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MemoryIdentityError";
  }
}

/** Base for pre-HTTP client/usage errors (no status code or response body). */
export class MemoryClientError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "MemoryClientError";
  }
}

/**
 * Raised when an operation is unavailable in the active transport mode.
 *
 * This is a client-side capability check raised before any HTTP request is
 * attempted, so it carries no status code or response body.
 */
export class MemoryNotSupportedError extends MemoryClientError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "MemoryNotSupportedError";
  }
}

/** Base for typed transport errors from the HTTP memory transports. */
export class MemoryAPIError extends Error {
  readonly status: number | null;
  readonly code: string | null;
  readonly responseText: string | null;

  constructor(
    message: string,
    opts: {
      status?: number | null;
      code?: string | null;
      responseText?: string | null;
    } = {},
  ) {
    // Keep `message` and the code annotation consistent: the annotated form is
    // what `super` records, so `err.message`/`toString()`/stack all surface the
    // code. `.code` remains separately available for programmatic branching.
    super(opts.code ? `${message} (code=${opts.code})` : message);
    this.name = "MemoryAPIError";
    this.status = opts.status ?? null;
    this.code = opts.code ?? null;
    this.responseText = opts.responseText ?? null;
  }
}

/** Raised for 401 / 403 responses. */
export class MemoryAuthError extends MemoryAPIError {
  constructor(
    message: string,
    opts?: ConstructorParameters<typeof MemoryAPIError>[1],
  ) {
    super(message, opts);
    this.name = "MemoryAuthError";
  }
}

/** Raised when the project's memory runtime is not yet reachable. */
export class MemoryNotProvisionedError extends MemoryAPIError {
  constructor(
    message: string,
    opts?: ConstructorParameters<typeof MemoryAPIError>[1],
  ) {
    super(message, opts);
    this.name = "MemoryNotProvisionedError";
  }
}

/** Raised for 4xx responses other than auth / not-provisioned. */
export class MemoryBadRequestError extends MemoryAPIError {
  constructor(
    message: string,
    opts?: ConstructorParameters<typeof MemoryAPIError>[1],
  ) {
    super(message, opts);
    this.name = "MemoryBadRequestError";
  }
}

/**
 * Raised when a core-loop request 404s, hinting a route-shape mismatch.
 *
 * The SDK selects the route shape from `projectId` presence (set => project-scoped
 * Gateway routes; empty => flat OE routes). A 404 on a core-loop POST most often
 * means that shape does not match the backend the `baseUrl` points at, so this
 * carries a directional, actionable hint rather than the opaque 404. It extends
 * `MemoryBadRequestError` so existing 4xx handling still catches it.
 */
export class MemoryRouteNotFoundError extends MemoryBadRequestError {
  constructor(
    message: string,
    opts?: ConstructorParameters<typeof MemoryAPIError>[1],
  ) {
    super(message, opts);
    this.name = "MemoryRouteNotFoundError";
  }
}

/**
 * Raised for 5xx responses, and for success responses whose body is unparseable
 * or has an unexpected shape.
 */
export class MemoryServerError extends MemoryAPIError {
  constructor(
    message: string,
    opts?: ConstructorParameters<typeof MemoryAPIError>[1],
  ) {
    super(message, opts);
    this.name = "MemoryServerError";
  }
}

/** Raised when the transport fails to reach the memory backend (connect/timeout). */
export class MemoryConnectionError extends MemoryAPIError {
  constructor(
    message: string,
    opts?: ConstructorParameters<typeof MemoryAPIError>[1],
  ) {
    super(message, opts);
    this.name = "MemoryConnectionError";
  }
}
