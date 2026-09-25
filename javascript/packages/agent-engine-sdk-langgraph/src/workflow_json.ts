/** JSON and protobuf boundary helpers for durable workflow state. */

import { fromJson, toJson, type JsonObject } from "@bufbuild/protobuf";
import { ValueSchema, type Value } from "@bufbuild/protobuf/wkt";
import { JsonValueSchema, type JsonValue } from "@mongodb-js/agent-engine-sdk";

export function serializeJsonValue(value: unknown, path: string): JsonValue {
  let parsed;
  try {
    parsed = JsonValueSchema.safeParse(value);
  } catch (error) {
    throw new TypeError(`${path} is not JSON-serializable`, { cause: error });
  }
  if (!parsed.success) {
    throw new TypeError(`${path} is not JSON-serializable`);
  }
  return parsed.data;
}

export function serializeJsonObject(value: unknown, path: string): JsonObject {
  const serialized = serializeJsonValue(value, path);
  if (
    serialized === null ||
    Array.isArray(serialized) ||
    typeof serialized !== "object"
  ) {
    throw new TypeError(`${path} must be an object`);
  }
  return serialized;
}

export function serializeJsonObjectArray(
  value: unknown,
  path: string,
): JsonObject[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${path} must be an array`);
  }
  return value.map((item, index) =>
    serializeJsonObject(item, `${path}[${index}]`),
  );
}

export function requireJsonObject(value: unknown, path: string): JsonObject {
  const parsed = JsonValueSchema.safeParse(value);
  if (
    !parsed.success ||
    parsed.data === null ||
    Array.isArray(parsed.data) ||
    typeof parsed.data !== "object"
  ) {
    throw new TypeError(`${path} must be a JSON object`);
  }
  return parsed.data;
}

export function protoValueFromUnknown(value: unknown, path: string): Value {
  return fromJson(ValueSchema, serializeJsonValue(value, path));
}

export function jsonValueFromProto(value: Value | undefined): JsonValue {
  return value === undefined ? null : (toJson(ValueSchema, value) as JsonValue);
}
