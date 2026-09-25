# Table of Contents

* [agent\_engine\_sdk\_adk.runtime](#agent_engine_sdk_adk.runtime)
  * [App](#agent_engine_sdk_adk.runtime.App)
    * [\_\_init\_\_](#agent_engine_sdk_adk.runtime.App.__init__)
    * [memory](#agent_engine_sdk_adk.runtime.App.memory)
    * [get\_current\_user\_id](#agent_engine_sdk_adk.runtime.App.get_current_user_id)
    * [tools](#agent_engine_sdk_adk.runtime.App.tools)
    * [tool](#agent_engine_sdk_adk.runtime.App.tool)
    * [entrypoint](#agent_engine_sdk_adk.runtime.App.entrypoint)
    * [llm](#agent_engine_sdk_adk.runtime.App.llm)
    * [get\_agent](#agent_engine_sdk_adk.runtime.App.get_agent)
    * [run](#agent_engine_sdk_adk.runtime.App.run)
* [agent\_engine\_sdk\_adk.agent](#agent_engine_sdk_adk.agent)
  * [ADKBaseAgent](#agent_engine_sdk_adk.agent.ADKBaseAgent)
* [agent\_engine\_sdk\_adk.runner](#agent_engine_sdk_adk.runner)
  * [DurableADKRunner](#agent_engine_sdk_adk.runner.DurableADKRunner)
    * [rewind\_async](#agent_engine_sdk_adk.runner.DurableADKRunner.rewind_async)
* [agent\_engine\_sdk\_adk.rewind](#agent_engine_sdk_adk.rewind)
  * [RewindBranchError](#agent_engine_sdk_adk.rewind.RewindBranchError)
  * [create\_rewind\_branch](#agent_engine_sdk_adk.rewind.create_rewind_branch)

Google ADK adapter runtime for Atlas Agent Engine.

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

<a id="agent_engine_sdk_adk.runtime.App.tool"></a>

#### tool

```python
def tool(
    is_local: bool = True,
    *,
    network: list[str] | None = None,
    timeout: int = 30,
    redact_fields: list[str] | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]
```

Register a tool function with the platform.

<a id="agent_engine_sdk_adk.runtime.App.entrypoint"></a>

#### entrypoint

```python
def entrypoint(fn: Callable[..., Any]) -> Callable[..., Any]
```

Mark the ADK agent builder function.

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

<a id="agent_engine_sdk_adk.runtime.App.get_agent"></a>

#### get\_agent

```python
def get_agent(callbacks: list[Any] | None = None) -> Any
```

Build and return an ADKBaseAgent wrapping the registered entrypoint.

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
