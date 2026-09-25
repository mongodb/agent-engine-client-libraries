/**
 * URL path helpers for outbound platform HTTP calls.
 *
 * Shared choke point for percent-encoding one path segment. Framework adapters
 * must route every interpolated identifier through {@link quotePathSegment}
 * instead of hand-rolling quoting: bare `encodeURIComponent` leaves
 * `.`/`..` untouched and string-concatenated identifiers forge paths.
 *
 * Port of Python's `agent_engine_runner_shared/http_path.py`.
 */

const MAX_DECODE_ROUNDS = 8;

function fullyDecode(value: string): string {
  let decoded = value;
  for (let round = 0; round < MAX_DECODE_ROUNDS; round++) {
    let next: string;
    try {
      next = decodeURIComponent(decoded);
    } catch {
      // Malformed `%` sequences and invalid UTF-8 escapes are rejected the
      // same as Python's `unquote(..., errors="strict")`.
      throw new Error("URL path segment is invalid");
    }
    if (next === decoded) return decoded;
    decoded = next;
  }
  throw new Error("URL path segment is invalid");
}

/**
 * Percent-encode one path segment; reject separators and `.` / `..`.
 *
 * Decode repeatedly before quoting so encoded, double-encoded, and mixed
 * traversal forms cannot survive a later routing decode.
 */
export function quotePathSegment(value: string): string {
  const decoded = fullyDecode(value);
  if (
    !decoded ||
    decoded === "." ||
    decoded === ".." ||
    /[/\\?#]/.test(decoded)
  ) {
    throw new Error("URL path segment is invalid");
  }
  // `encodeURIComponent` keeps `! ' ( ) *` unescaped; Python's
  // `quote(decoded, safe="")` does not. Match it so both runtimes produce
  // byte-identical paths.
  return encodeURIComponent(decoded)
    .replaceAll("!", "%21")
    .replaceAll("'", "%27")
    .replaceAll("(", "%28")
    .replaceAll(")", "%29")
    .replaceAll("*", "%2A");
}
