import { fromJson, toJson, type MessageShape } from "@bufbuild/protobuf";
import type { GenEnum, GenMessage } from "@bufbuild/protobuf/codegenv2";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";

import * as activity from "../../src/generated/workflow/v1/activity_pb.js";
import * as common from "../../src/generated/workflow/v1/common_pb.js";
import * as runtime from "../../src/generated/workflow/v1/runtime_pb.js";
import * as state from "../../src/generated/workflow/v1/state_pb.js";

const fixturePath = (name: string) =>
  fileURLToPath(
    new URL(
      `../../../../../../../proto/workflow/v1/testdata/protojson/${name}`,
      import.meta.url,
    ),
  );

function isMessageSchema(value: unknown): value is GenMessage<MessageShape> {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "message"
  );
}

function isEnumSchema(value: unknown): value is GenEnum<number> {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    value.kind === "enum"
  );
}

const wireModules = [activity, common, runtime, state];
const wireSchemas = wireModules.flatMap((module) =>
  Object.values(module).filter(isMessageSchema),
);
const wireEnumSchemas = wireModules.flatMap((module) =>
  Object.values(module).filter(isEnumSchema),
);

describe("workflow.v1 ProtoJSON profile", () => {
  test("covers every generated wire message with populated fixtures", () => {
    const expected = JSON.parse(
      readFileSync(fixturePath("wire_messages.json"), "utf8"),
    ) as Record<string, Record<string, unknown>[]>;
    expect(wireSchemas.map((schema) => schema.typeName).sort()).toEqual(
      Object.keys(expected).sort(),
    );
    for (const schema of wireSchemas) {
      const fixtures = expected[schema.typeName];
      expect(fixtures.length).toBeGreaterThan(0);
      for (const fixture of fixtures) {
        expect(
          toJson(
            schema,
            fromJson(schema, fixture, { ignoreUnknownFields: false }),
            { useProtoFieldName: true },
          ),
        ).toEqual(fixture);
      }
      expect(toJson(schema, fromJson(schema, {}))).toEqual({});
    }
  });

  test("emits snake-case names, enum names, quoted int64, and no defaults", () => {
    const expected = (
      JSON.parse(
        readFileSync(fixturePath("wire_messages.json"), "utf8"),
      ) as Record<string, Record<string, unknown>[]>
    )["mongodb.agentic.workflow.v1.AttemptContext"][0];
    const message = fromJson(runtime.AttemptContextSchema, {
      attemptId: "attempt-1",
      fencingToken: "7",
      ownerId: "aer-1",
      replayMode: true,
      workflowIdentity: {
        tenantScope: {
          orgId: "org-1",
          projectId: "project-1",
          workspaceId: "workspace-1",
        },
        sessionId: "session-1",
        executionId: "execution-1",
      },
      declaration: {
        workflowName: "insurance-agent",
        workflowVersion: "1",
        adapterName: "langgraph",
        adapterVersion: "1",
        memoryEnabled: true,
      },
      heartbeatIntervalMs: "5000",
      previousState: {
        properties: { claim_count: 1 },
      },
      branchLineage: {
        sourceWorkflowIdentity: {
          tenantScope: {
            orgId: "org-1",
            projectId: "project-1",
            workspaceId: "workspace-1",
          },
          sessionId: "source-session",
          executionId: "source-execution",
        },
        sourceStepOrdinal: "4",
        sourceStateHash: "sha256:source",
      },
    });

    expect(
      toJson(runtime.AttemptContextSchema, message, {
        useProtoFieldName: true,
      }),
    ).toEqual(expected);
    expect(
      fromJson(
        runtime.AttemptContextSchema,
        {
          attemptId: "attempt-1",
          fencingToken: "7",
          ownerId: "aer-1",
          replayMode: true,
        },
        { ignoreUnknownFields: false },
      ).fencingToken,
    ).toBe(7n);
    expect(
      toJson(
        runtime.AttemptContextSchema,
        fromJson(runtime.AttemptContextSchema, {}),
        {
          useProtoFieldName: true,
        },
      ),
    ).toEqual({});
  });

  test("covers every generated wire enum", () => {
    const expected = JSON.parse(
      readFileSync(fixturePath("wire_enums.json"), "utf8"),
    );
    const actual = Object.fromEntries(
      wireEnumSchemas.map((schema) => [
        schema.typeName,
        schema.values.map((value) => value.name),
      ]),
    );
    expect(actual).toEqual(expected);
  });

  test("rejects unknown mutation fields and tolerates them for reads", () => {
    expect(() =>
      fromJson(
        activity.ActivityCommandSchema,
        { activity_name: "tool", future_field: true },
        { ignoreUnknownFields: false },
      ),
    ).toThrow();

    expect(
      fromJson(
        runtime.AttemptContextSchema,
        { attempt_id: "attempt-1", future_field: true },
        { ignoreUnknownFields: true },
      ).attemptId,
    ).toBe("attempt-1");
  });
});
