/**
 * Serialization helpers shared by the layout and the stdout-capture stream:
 * a crash-proof JSON stringifier and a UTF-8-aware truncator.
 */

/**
 * Stringify the structured-logging envelope with the same robustness Python gets from
 * `json.dumps(entry, default=str)`. Two failure modes the bare
 * `JSON.stringify` would otherwise expose:
 *
 *   - `TypeError: Do not know how to serialize a BigInt` — any extras
 *     value that happens to be a BigInt (e.g. nanosecond timings,
 *     64-bit IDs) crashes the appender for the whole record.
 *   - `TypeError: Converting circular structure to JSON` — a caller
 *     accidentally passes a node from a graph (Error.cause chains in
 *     newer Node, langgraph state) and takes down logging for that line.
 *
 * The replacer coerces BigInt to a decimal string and replaces objects that
 * are their own ancestor (a true reference cycle) with the sentinel
 * `"[Circular]"`. Tracking the ancestor chain rather than a flat seen-set
 * means a DAG — the same object referenced from multiple siblings, e.g. an
 * array of identical items — serializes in full instead of being falsely
 * truncated to `"[Circular]"`. A fresh `ancestors` stack per call means
 * independent records can't poison each other.
 */
export function jsonSafeStringify(value: unknown): string {
  const ancestors: object[] = [];
  return JSON.stringify(value, function (this: unknown, _key, v) {
    if (typeof v === "bigint") return v.toString();
    if (typeof v !== "object" || v === null) return v;
    // `this` is the object `v` was reached through. Unwind ancestors that are
    // no longer on the path to `v` so only genuine back-references remain.
    while (ancestors.length > 0 && ancestors[ancestors.length - 1] !== this) {
      ancestors.pop();
    }
    if (ancestors.includes(v)) return "[Circular]";
    ancestors.push(v);
    return v;
  });
}

export function truncateUtf8(text: string, maxBytes: number): string {
  // Encode to bytes for the size check so the budget reflects what Fluent
  // Bit actually ships — slicing the string by char count would let a
  // non-ASCII traceback blow past the limit. Walk back from `maxBytes` to
  // the last UTF-8 sequence boundary before decoding: Node's
  // `Buffer.toString("utf-8")` REPLACES a partial sequence at the cut
  // point with U+FFFD (3 bytes each), which would expand the result past
  // the budget. Walking back over continuation bytes (`10xxxxxx`,
  // i.e. `byte & 0xc0 === 0x80`) drops the partial sequence so the
  // decoded body is strictly `<= maxBytes`. Mirrors Python's
  // `errors="ignore"` decode at structured_logging.py:146.
  const buf = Buffer.from(text, "utf-8");
  if (buf.length <= maxBytes) return text;
  let cut = maxBytes;
  while (cut > 0 && ((buf[cut] ?? 0) & 0xc0) === 0x80) cut--;
  return buf.subarray(0, cut).toString("utf-8") + "...(truncated)";
}
