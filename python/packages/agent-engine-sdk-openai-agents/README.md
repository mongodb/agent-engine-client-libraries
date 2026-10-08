# Agent Engine SDK OpenAI Agents

[![License](https://img.shields.io/badge/license-Apache-2.0-blue.svg)](LICENSE)

OpenAI Agents SDK adapter for the MongoDB Atlas Agent Engine SDK.

> **Status:** under construction. `framework: openai-agents` is not yet
> accepted by the CLI or the build service.

The adapter runs a native OpenAI Agents `Agent` through the SDK's own
`Runner` while the Orchestration Engine owns durable execution. The adapter
keeps no session database and never uses OpenAI Sessions, Conversations, or
`previous_response_id` as continuation state.

## Building an agent

```python
from agents import Agent, ModelSettings
from agent_engine_sdk_openai_agents import App

app = App(app_name="support")


@app.tool()
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    ...


@app.entrypoint
def build() -> Agent:
    return Agent(
        name="support",
        instructions="Help with orders.",
        model=app.llm("gpt-4.1", settings=ModelSettings(temperature=0)),
        tools=app.tools(),
    )
```

Use the model returned by `app.llm(...)` and the tools returned by
`app.tools()`; anything else fails the turn before any model or tool work.
Pass a configured `agents.Model` to `app.llm` to use a custom OpenAI client.
Tool parameters must be JSON values that can be passed by name (no
positional-only or `*args` parameters, and none named `tool_call_id`), because remote tools receive their
arguments as named JSON values. Set a tool's timeout with
`@app.tool(timeout=...)`; a native `timeout_seconds` on an issued tool fails
the turn. A non-string tool result, including nested models,
dataclasses, enums, and datetimes, reaches the model as JSON text. The text is
canonical, so a replayed turn sends the model the same text: keys are sorted
and a whole-number float such as `164.0` is written `164`. An integer beyond
2**53 in a tool's result or arguments fails the call, because the durable
record cannot keep it exactly; pass such a value as a string.

Each invocation runs one native `Runner` turn and commits it to the
Orchestration Engine before returning the result. A failed tool call fails
the turn with the error the platform's tool path reports, such as a tool
execution error carrying the tool's message or a policy denial; a cancelled or
user-stopped call fails it too. Nothing is committed in any of these cases.
When the model calls several tools at once, they run concurrently. Each call
keeps the Orchestration Engine activity of its call id, and their results reach
the model in the order the model made the calls, whichever finishes first. If
one of them waits for approval, the calls that ran before the wait keep their
outputs ahead of the approved call's, as the SDK records them. When one call
fails, the turn fails once the calls still running beside it have finished.
MCP servers, guardrails, agent hooks, hosted prompts, dynamic instructions,
and a custom `tool_use_behavior` are not supported yet.

## Handoffs

Use the SDK's own `handoffs=[...]`, with other agents or `agents.handoff(...)`.
The runtime reads the whole agent graph the entrypoint returns before each
turn and holds every reachable agent to the same rules as the first: its model
comes from `app.llm(...)` (give each agent's model its own `llm_id`) and its
tools from `app.tools()`, from which each agent can select its own by name.
Agent names must be unique.

A handoff changes which agent is active; the turn stays one execution. The
committed state records the active agent, so the next turn, also after a
restart, starts with the agent that ended the last one. An approval requested
by a handed-off agent suspends and resumes in that agent.

`agents.handoff(...)` may override the tool name and description. Agents may
hand off to each other in a cycle. An `input_filter`, `nest_handoff_history`,
an `on_handoff` callback, an `input_type`, a callable `is_enabled`, and a
hand-built `Handoff` fail the turn before any model or tool work. So does a
handoff on any agent whose tool name matches another handoff's or one of that
agent's tools: the SDK would otherwise keep one and drop the other.

## Agents as tools

Use the SDK's own `Agent.as_tool(...)` and put the result in another agent's
`tools`. The nested agent follows the same rules as every other agent in the
graph. Each call runs the nested agent as its own operation: its model and
tool calls are Orchestration Engine activities under a path named after the
tool, and repeated or concurrent calls to the same agent tool stay distinct,
in the order the model made them.

The calling agent's model sees only the tool's answer, and only the calling
conversation is committed; a replacement attempt replays the nested run from
its recorded activities. The caller's stream carries the nested agent's text
between `subagent_start` and `subagent_end` events, tagged with the tool name
and the call id, apart from the calling agent's reply. A nested run that fails
fails the turn.

`max_turns`, `custom_output_extractor`, and structured input are supported. A
replacement attempt calls the extractor and a structured input's
`input_builder` again, so both must return the same value for the same input
and have no side effects; OE fails the turn when a replayed model call differs
from the recorded one.

An `on_stream` callback, `hooks`, a `run_config`, a `session`, provider
continuation ids, a callable `needs_approval` or `is_enabled`, tool guardrails,
a native `timeout_seconds`, and an agent that can reach itself through its own
agent tool fail the turn before any model or tool work. So does a tool that
needs approval anywhere a nested agent can reach, which is not supported yet;
approvals in the calling agent work as usual.

## Approvals

Register a tool with `@app.tool(needs_approval=True)` to have a person approve
each call. The argument is the SDK's own `function_tool(needs_approval=...)`,
passed through to the `FunctionTool` that `app.tools()` returns; setting
`needs_approval` on that returned tool works too. When the model calls a tool
that needs approval, the turn suspends: the invocation returns
`status: suspended` with one interrupt per call waiting for approval, in the
order the model made the calls, each naming the tool and its arguments. Calls
in the same response that need no approval run first. Resume through the normal
invoke endpoint on the same session, with `resume_map` answering every
interrupt id at once:

```json
{"resume_map": {"<interrupt id>": {"confirmed": true}}}
```

`interrupts` lists every pending approval. `suspend_payload` carries only the
first one's value, for callers that read a single payload.

`{"confirmed": false, "rejection_message": "..."}` rejects the call; the model
sees the rejection message instead of a tool result, as with the SDK's own
`reject(..., rejection_message=...)`. Without a
`rejection_message` the model sees the SDK's default rejection text; an empty
one is refused. An approval
carries no message. The Orchestration Engine records the wait
and its answer. Resuming replays the turn from the recorded model calls, applies
the answer through the SDK's own `RunState`, and runs only what remains, so a
tool never runs before its approval, prefix output is not repeated, and a
restart while waiting loses nothing. A resume that leaves any interrupt
unanswered is refused; `needs_approval` must be `True` or `False`, not a
callable.

## Durable state

Each committed turn stores the Runner's complete input list under the
`__agent_engine_openai_agents__` state property, together with the app and
session it belongs to and the name of the agent that ended the turn. The next
turn continues from that list with that agent. Committed
messages are an explicit-field projection for history readers; they are not
read back.

Supported items are text messages, function calls, and text function outputs.
The calls one model response made come together, followed by exactly one
output per call before anything else. Any other item, any state that belongs
to a different app or session, and an active agent the app no longer has fail
the turn before new model or tool work starts.

## Model calls

Every model call the native `Runner` makes becomes an Orchestration Engine
LLM activity. The agent's model answers from the recorded activity, and the
model itself runs in the Tool Pod. A replacement attempt replays the recorded
response and gives the `Runner` identical output items.

Only per-call settings the platform carries cross that boundary: `max_tokens`,
`top_p`, `frequency_penalty`, `presence_penalty`, `timeout`,
`parallel_tool_calls`, `reasoning.effort`, `verbosity`, and a named
`tool_choice`. Provider settings such as `temperature` belong to the model you
register, and the Tool Pod applies them. A registered model cannot carry a
`tool_choice` or `timeout` (set them on the agent), or `extra_body` / `extra_args` fields that make the provider
continue a stored conversation (`previous_response_id`, `conversation`,
`conversation_id`, `prompt`) or that override what the adapter builds for
each call (such as `tools`, `input`, `instructions`, or the output format), or that set
`reasoning`, `reasoning_effort`, or `verbosity` as raw fields (use `ModelSettings.reasoning` and
`ModelSettings.verbosity`). Native tracing, provider-hosted
continuation, and stop sequences fail before the call is sent.

### Reasoning

Set `reasoning=Reasoning(effort=...)` and `verbosity` on the registered model,
on the agent, or both; per-call values win over the registration. Prefer the
registration. The SDK cannot tell an agent's explicit settings from its own
defaults: when `Agent(model_settings=...)` equals the SDK's default settings
(currently `Reasoning(effort="none")` with `verbosity="low"`), the SDK clears
them at construction and the registered values apply. To use exactly that pair
per call, assign it after construction (`agent.model_settings = ...`). Other
reasoning settings, such as `reasoning.summary`, fail before the call is sent.
Reasoning-token counts are reported in the model call's usage. Providers may
limit these settings per model and API: for example, a Chat Completions
reasoning model can refuse function tools unless the effort is `none`. The
provider's refusal fails the model call.

A durable run does not carry a model's hidden reasoning from one model call to
the next. The provider's encrypted reasoning items are dropped in the Tool Pod
and never stored, so each call, including each step of a tool loop, reasons
from the durable conversation alone. If an agent needs a plan that persists
across calls, it should record the plan explicitly, for example in its own
messages or in Memory, rather than rely on the model's hidden reasoning.

The Tool Pod retries transient provider failures before any output streams.
The SDK's own `ModelSettings.retry` is rejected: on the agent it would make
the `Runner` issue extra model activities, and on a registered model it would
have no effect. A model stream that ends without a completed response fails
the call rather than being recorded as a truncated answer.
