# Table of Contents

* [agent\_engine\_sdk\_adk.runtime](#agent_engine_sdk_adk.runtime)
  * [LLMRegistrationDriftError](#agent_engine_sdk_adk.runtime.LLMRegistrationDriftError)
  * [App](#agent_engine_sdk_adk.runtime.App)
    * [\_\_init\_\_](#agent_engine_sdk_adk.runtime.App.__init__)
    * [memory](#agent_engine_sdk_adk.runtime.App.memory)
    * [get\_current\_user\_id](#agent_engine_sdk_adk.runtime.App.get_current_user_id)
    * [tools](#agent_engine_sdk_adk.runtime.App.tools)
    * [tool](#agent_engine_sdk_adk.runtime.App.tool)
    * [entrypoint](#agent_engine_sdk_adk.runtime.App.entrypoint)
    * [llm](#agent_engine_sdk_adk.runtime.App.llm)
    * [build\_graph](#agent_engine_sdk_adk.runtime.App.build_graph)
    * [get\_agent](#agent_engine_sdk_adk.runtime.App.get_agent)
    * [run](#agent_engine_sdk_adk.runtime.App.run)
* [agent\_engine\_sdk\_adk.agent](#agent_engine_sdk_adk.agent)
  * [ADKBaseAgent](#agent_engine_sdk_adk.agent.ADKBaseAgent)
    * [cancelled\_history\_scope](#agent_engine_sdk_adk.agent.ADKBaseAgent.cancelled_history_scope)
* [agent\_engine\_sdk\_adk.runner](#agent_engine_sdk_adk.runner)
  * [DurableADKRunner](#agent_engine_sdk_adk.runner.DurableADKRunner)
    * [rewind\_async](#agent_engine_sdk_adk.runner.DurableADKRunner.rewind_async)
* [agent\_engine\_sdk\_adk.rewind](#agent_engine_sdk_adk.rewind)
  * [RewindBranchError](#agent_engine_sdk_adk.rewind.RewindBranchError)
  * [create\_rewind\_branch](#agent_engine_sdk_adk.rewind.create_rewind_branch)

Google ADK adapter runtime for Atlas Agent Engine.

<a id="agent_engine_sdk_adk.runtime.LLMRegistrationDriftError"></a>

## LLMRegistrationDriftError

```python
class LLMRegistrationDriftError(RuntimeError)
```

A per-execution build registered a different LLM id set.

Distinct from an ordinary build failure so the Tool Pod can tell a registry
that has drifted from one that merely failed to build: a drifted registry is
not a usable partial discovery. The marker attribute lets runner-shared
recognise it without importing this SDK, which it is not allowed to do.

<a id="agent_engine_sdk_adk.runtime.App"></a>

## App

```python
class App(BaseApp)
```

Google ADK SDK for Atlas Agent Engine.

**Example**:


```python
from agent_engine_sdk_adk import App
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

app = App(app_name="My Agent")


@app.tool()
def my_tool(query: str) -> str:
    """Search the database."""
    return "result"


@app.entrypoint
def build_agent():
    return LlmAgent(
        model=app.llm(Gemini(model="gemini-3-flash-preview")),
        tools=app.tools(),
        name="my_agent",
    )
```

<a id="agent_engine_sdk_adk.runtime.App.__init__"></a>

#### \_\_init\_\_

```python
def __init__(app_name: str,
             app_version: str = "1.0.0",
             mongodb_uri: str | None = None,
             database_name: str | None = None,
             enable_tracing: bool | None = None,
             traces_collection_name: str = "traces",
             enable_memory: bool | None = None,
             org_id: str | None = None) -> None
```

Initialize the App.

**Arguments**:

- `app_name` - Application name
- `app_version` - Application version
- `mongodb_uri` - MongoDB URI
- `database_name` - Database name
- `enable_tracing` - Deprecated. Tracing is always enabled.
- `traces_collection_name` - Collection name for trace storage
- `enable_memory` - Deprecated. Use agent.yaml features.memory instead.
- `org_id` - Deprecated and ignored. The org is taken from the
  ``ORG_ID`` environment variable, which the platform injects.
  Passing this argument raises a DeprecationWarning and will be
  removed in a future release.

<a id="agent_engine_sdk_adk.runtime.App.memory"></a>

#### memory

```python
@property
def memory() -> Memory
```

Return the unified Memory facade over app-bound adapters.

The return type is ``agent_engine_sdk_memory.Memory``. Methods return
typed results (``Create*Result``, ``MemoryChunk``, ``ContextResponse``).
Identity is supplied via per-call args, ``bind``, or ambient
execution contextvars.

<a id="agent_engine_sdk_adk.runtime.App.get_current_user_id"></a>

#### get\_current\_user\_id

```python
def get_current_user_id() -> str | None
```

Return the user id of the currently-executing request, if any.

Backed by the request-scoped context agent-engine-runner-shared sets on the AER
before invoking the agent, so tools can attribute writes to the caller.

<a id="agent_engine_sdk_adk.runtime.App.tools"></a>

#### tools

```python
def tools() -> list[Any]
```

Return tool callables ready for an ADK agent builder.

In AER mode, wrap each tool with SecureToolWrapper. Authors
apply native ADK HITL constructors to these returned callables:
``FunctionTool(fn, require_confirmation=True)`` and
``LongRunningFunctionTool(fn)``. Do not wrap the raw ``@app.tool``
function; that has not gone through the secure wrapper yet.

Tools discovered from ``agent.yaml`` ``mcp.servers`` entries are
returned as ready-made ADK tools carrying the server's JSON schema;
their calls also route through the secure wrapper, with the MCP
identity OE forwards to the Tool Pod.

<a id="agent_engine_sdk_adk.runtime.App.tool"></a>

#### tool

```python
def tool(
    is_local: bool = True,
    *,
    network: list[str] | None = None,
    timeout: int = 30,
    redact_fields: list[str] | None = None,
    on_unknown_outcome: UnknownOutcomePolicy | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]
```

Register a tool function with the platform.

**Arguments**:

- `on_unknown_outcome` - What a retry does when the platform lost the
  runner while this tool was running in a durable workflow, so
  nobody knows whether it ran. ``"rerun"`` runs the tool again
  (at-least-once execution; declare it only for a tool that is
  safe to run twice). ``"fail"`` refuses the retry, so the tool
  never runs twice. Leave unset for the platform default.

<a id="agent_engine_sdk_adk.runtime.App.entrypoint"></a>

#### entrypoint

```python
def entrypoint(
        fn: Callable[..., Any] | None = None,
        *,
        load_strategy: Literal["global", "per-execution"] = "global") -> Any
```

Mark the ADK agent builder function.

Usable bare (``@app.entrypoint``) for the default behavior, or called
with a strategy:

```python
@app.entrypoint(load_strategy="per-execution")
async def build(input: AgentInput, ctx: RequestContext) -> BaseAgent:
    model = app.llm(Gemini(model="gemini-2.5-flash"), llm_id="primary")
    flag = await fetch_flag(ctx.user_id)
    return assemble_agent(model=model, rollout_enabled=flag)
```

``load_strategy="per-execution"`` runs the builder on every invocation
with the invocation's ``(input, ctx)``, so it can perform its own
request-scoped I/O before returning the agent. The builder's arity is
checked here, at decoration time, against the call the runtime will
actually make.

**Registering LLMs under ``per-execution``.** Register every ``llm_id``
the agent could ever use *before* any construction that depends on the
invocation context. The Tool Pod discovers an agent's registered ids by
building it with no invocation context, so a builder that needs context
cannot be built there; it keeps whatever was registered before such a
build failed. Reading context is fine, and reading it *before* the
registrations is what breaks discovery.

The registered set must also be the same on every invocation, because
the Tool Pod resolves a model by ``llm_id`` against a registry built
once per pod. Vary which id the agent *binds* per caller; do not vary
which ids are *registered*. A per-execution build that registers a
different set fails at the build, rather than mid-execution when the pod
cannot resolve the model:

```python
@app.entrypoint(load_strategy="per-execution")
async def build(input: AgentInput, ctx: RequestContext) -> BaseAgent:
    # Register every selectable model first...
    free = app.llm(Gemini(model="gemini-2.5-flash"), llm_id="free")
    premium = app.llm(Gemini(model="gemini-3-flash-preview"), llm_id="premium")
    # ...then read context, and bind one of them.
    tier = await fetch_tier(ctx.user_id)
    return LlmAgent(
        model=premium if tier == "premium" else free,
        tools=PREMIUM if tier == "premium" else FREE,
        name="my_agent",
    )
```

<a id="agent_engine_sdk_adk.runtime.App.llm"></a>

#### llm

```python
def llm(llm: BaseLlm, llm_id: str | None = None) -> BaseLlm
```

Wrap an ADK LLM for audited I/O through the Orchestration Engine.

**Arguments**:

- `llm` - ADK BaseLlm instance.
- `llm_id` - Unique identifier for this LLM.


**Returns**:

  SecureLlm in AER mode; the raw *llm* in TOOL mode.

<a id="agent_engine_sdk_adk.runtime.App.build_graph"></a>

#### build\_graph

```python
async def build_graph(agent_input: AgentInput | None = None,
                      ctx: RequestContext | None = None) -> Any
```

Build the agent for this invocation before ``get_agent``.

Returns ``None`` for a global agent, signalling the caller to use the
existing build path unchanged. For a per-execution agent, runs the
builder with ``(agent_input, ctx)`` -- awaiting it when the builder is
async -- and returns the agent.

The platform also builds the agent without an invocation to discover its
registered LLM ids. That call passes ``(None, None)``; a builder that
needs real context fails there, and the failure names the context-free
discovery rather than looking like a bug in the builder.

<a id="agent_engine_sdk_adk.runtime.App.get_agent"></a>

#### get\_agent

```python
def get_agent(callbacks: list[Any] | None = None, *, graph: Any = None) -> Any
```

Build and return an ADKBaseAgent wrapping the registered entrypoint.

``graph`` is an agent ``build_graph`` already built for this invocation.
It is only forwarded when present, so a caller using the existing path
keeps working.

<a id="agent_engine_sdk_adk.runtime.App.run"></a>

#### run

```python
def run(**kwargs: Any) -> None
```

Start the agent service.

Durable Google ADK adapter over the framework-neutral BaseAgent protocol.

``agent.py`` is the orchestrator. Durable setup lives on ``DurableSession``.
Suspend and resume: see the package README.

```
agent.py
├── execution_session.py   durable run (attempt, original input, Runner)
├── platform_session.py    scratch BaseSessionService + OE snapshot
├── stream.py              tokens / suspend / result
├── suspend.py             native ADK wait shapes
└── workflow.py            OE finalize_step / complete
```

<a id="agent_engine_sdk_adk.agent.ADKBaseAgent"></a>

## ADKBaseAgent

```python
class ADKBaseAgent(BaseAgent)
```

Run one ADK 2 agent with OE as its only durable authority.

<a id="agent_engine_sdk_adk.agent.ADKBaseAgent.cancelled_history_scope"></a>

#### cancelled\_history\_scope

```python
@contextmanager
def cancelled_history_scope(ctx: RequestContext,
                            input: AgentInput) -> Generator[CancelledHistory]
```

Expose cancellation capture to the shared AER reporting lifecycle.

The runner enters this scope before the durable attempt's work runs, so a
Stop that tears the run down mid-turn can still retain the stopped
transcript. The opener is the request message under the id the adapter
seeds; ``normalized_input`` may replace it once the hook input is known.

Public runner for Google ADK applications on MongoDB.

<a id="agent_engine_sdk_adk.runner.DurableADKRunner"></a>

## DurableADKRunner

```python
class DurableADKRunner()
```

Run durable ADK turns and create immutable rewind branches.

<a id="agent_engine_sdk_adk.runner.DurableADKRunner.rewind_async"></a>

#### rewind\_async

```python
async def rewind_async(
        *,
        user_id: str,
        session_id: str,
        rewind_before_invocation_id: str,
        run_config: RunConfig | None = None) -> SessionForkResponse
```

Fork a session immediately before one completed ADK invocation.

Unlike Google's native implementation, this leaves ``session_id``
unchanged and returns the new branch identity.

Turn-level Google ADK rewind backed by an OE branch.

<a id="agent_engine_sdk_adk.rewind.RewindBranchError"></a>

## RewindBranchError

```python
class RewindBranchError(RuntimeError)
```

OE could not create the requested rewind branch.

<a id="agent_engine_sdk_adk.rewind.create_rewind_branch"></a>

#### create\_rewind\_branch

```python
async def create_rewind_branch(
        *, execution_id: str, oe_url: str,
        rewind_before_invocation_id: str) -> SessionForkResponse
```

Create a new session from the turn before one ADK invocation.
