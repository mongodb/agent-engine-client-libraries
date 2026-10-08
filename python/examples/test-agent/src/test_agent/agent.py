from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Annotated, Any, cast
from uuid import UUID

from agent_engine_sdk.models import AgentInput, RequestContext
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agent_engine_runner_shared import get_current_custom_headers, get_current_payload
from agent_engine_sdk_langgraph import App

app = App(app_name="Test Agent")
LLM_PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}
DEFAULT_MODELS = {
    "openai": "gpt-5.6-luna",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3-flash-preview",
    "cerebras": "gpt-oss-120b",
}


def _llm_provider_api_keys() -> dict[str, str]:
    return {
        provider: os.environ.get(env_var, "")
        for provider, env_var in LLM_PROVIDER_ENV_VARS.items()
    }


def _selected_llm_provider(api_keys: dict[str, str]) -> str:
    for provider, api_key in api_keys.items():
        if api_key:
            return provider

    raise ValueError(
        "Unable to construct the test agent LLM. Configure one of the example "
        "provider environments or customize _build_llm()."
    )


@app.tool()
def get_weather(city: str) -> str:
    """Get current weather conditions for a city.

    Supports magic inputs for integration testing:
    - __fail__      raises an exception (tests error handling)
    - __slow__      sleeps 30s (tests timeout handling)
    - __slow__:N    sleeps N seconds
    - __review__    triggers HITL suspend (tests suspend/resume)
    """
    if city == "__fail__":
        raise RuntimeError("Deliberate test failure")

    if city.startswith("__slow__"):
        parts = city.split(":")
        seconds = int(parts[1]) if len(parts) > 1 else 30
        time.sleep(seconds)
        return f"Slow response after {seconds}s"

    if city == "__review__":
        return app.suspend(
            reason="awaiting_human_review",
            context={"city": city},
        )

    # Store last queried city for persistence testing.
    # Wrapped in try/except because the tool pod may not have memory configured
    # (MONGODB_URI not set). Integration tests verify the write via the AER pod.
    try:
        app.memory.save_semantic(
            text=city,
            label="last_queried_city",
            user_id="test-user",
        )
    except Exception:
        pass

    return f"{city}: 22°F"


@app.tool()
def get_custom_headers() -> str:
    """Return the custom headers forwarded from the caller as a JSON string.

    Used by integration tests to verify that custom HTTP headers
    (X-Mdb-Agent-Engine-Custom-*) are available inside the agent.
    """
    headers = get_current_custom_headers()
    return json.dumps(headers)


@app.tool()
def get_payload() -> str:
    """Return the caller-provided invocation payload as a JSON string.

    Used by integration tests to verify that the request payload (the JSON
    body sent alongside `message`, e.g. extra context) is available inside
    the agent via agent_engine_runner_shared.get_current_payload().
    """
    return json.dumps(get_current_payload())


@app.prepare_agent_input
def prepare_agent_input(input: AgentInput, ctx: RequestContext) -> dict[str, Any]:
    """Build the graph's initial messages from the invocation payload.

    Mirrors the default message-wrapping so ordinary invocations are
    unaffected, and additionally folds a caller payload's ``prompt_context``
    into a leading message. Integration tests use this to verify the payload
    reaches the agent's initial input, not only tool calls.
    """
    payload = input.payload if isinstance(input.payload, dict) else {}
    messages: list[Any] = []
    prompt_context = payload.get("prompt_context")
    if isinstance(prompt_context, str) and prompt_context:
        messages.append(HumanMessage(content=prompt_context))
    messages.append(HumanMessage(content=str(payload.get("message", ""))))
    return {"messages": messages}


def _normalized_openai_base_url(base_url: str) -> str:
    normalized_base_url = base_url.rstrip("/")
    return normalized_base_url.removesuffix("/chat/completions").rstrip("/")


def _configured_openai_base_url() -> str | None:
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    return base_url or None


def _normalized_anthropic_base_url(base_url: str) -> str:
    # The Anthropic SDK appends /v1/messages itself; a base URL that already
    # carries the suffix would 404 on every call (AP-4544).
    normalized_base_url = base_url.rstrip("/")
    return normalized_base_url.removesuffix("/v1/messages").rstrip("/")


def _configured_anthropic_base_url() -> str | None:
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    return base_url or None


def _default_headers_with_foundry_api_key(
    existing_headers: Any,
    openai_key: str,
) -> dict[str, Any]:
    headers = dict(existing_headers) if isinstance(existing_headers, dict) else {}
    headers["api-key"] = openai_key
    return headers


def _is_foundry_base_url(base_url: str) -> bool:
    normalized = base_url.lower()
    return "foundry" in normalized or "azure" in normalized


def _normalized_foundry_base_url(base_url: str) -> str:
    return base_url.split("/v1")[0].rstrip("/") + "/v1"


def _build_llm():
    api_keys = _llm_provider_api_keys()
    selected_provider = _selected_llm_provider(api_keys)
    if selected_provider == "openai":
        from langchain_openai import ChatOpenAI

        llm_options: dict[str, Any] = {}
        openai_base_url = _configured_openai_base_url()
        if openai_base_url:
            uses_foundry_header = _is_foundry_base_url(openai_base_url)
            if uses_foundry_header:
                llm_options["base_url"] = _normalized_foundry_base_url(openai_base_url)
                llm_options["default_headers"] = _default_headers_with_foundry_api_key(
                    llm_options.get("default_headers"),
                    api_keys["openai"],
                )
            else:
                llm_options["base_url"] = _normalized_openai_base_url(openai_base_url)
        return cast(Any, ChatOpenAI)(
            api_key=api_keys["openai"],
            model=DEFAULT_MODELS["openai"],
            **llm_options,
        )

    if selected_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm_options = {}
        anthropic_base_url = _configured_anthropic_base_url()
        if anthropic_base_url:
            llm_options["base_url"] = _normalized_anthropic_base_url(anthropic_base_url)
        return cast(Any, ChatAnthropic)(
            api_key=api_keys["anthropic"],
            model_name=DEFAULT_MODELS["anthropic"],
            **llm_options,
        )

    if selected_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return cast(Any, ChatGoogleGenerativeAI)(
            api_key=api_keys["gemini"],
            model=DEFAULT_MODELS["gemini"],
        )

    if selected_provider == "cerebras":
        from langchain_cerebras import ChatCerebras

        return cast(Any, ChatCerebras)(
            api_key=api_keys["cerebras"],
            model=DEFAULT_MODELS["cerebras"],
        )

    raise ValueError(
        "Unable to construct the test agent LLM. Configure one of the example "
        "provider environments or customize _build_llm()."
    )


@app.entrypoint
def build_agent():
    """Two-LLM graph that also exercises tool-calling.

    Node ``rephrase`` uses ``llm_id='primary'`` to rephrase the user's question;
    node ``answer`` uses ``llm_id='secondary'`` and binds the registered tools
    so it can call ``get_custom_headers`` / ``get_weather`` when the prompt
    asks for them. The distinct llm_id values exercise the named-LLM registry
    lookup on the tool pod end-to-end. Tool-call messages are routed through
    ``ToolNode`` and back to ``answer`` until the model returns a final reply.
    """
    llm_primary = app.llm(_build_llm(), llm_id="primary")
    llm_secondary = app.llm(_build_llm(), llm_id="secondary")
    tools = app.get_tools()
    bound_secondary = llm_secondary.bind_tools(app.get_tool_schemas())

    def rephrase(state: MessagesState):
        prompt = state["messages"] + [
            HumanMessage(content="Rephrase the above question in one sentence.")
        ]
        return {"messages": [llm_primary.invoke(prompt)]}

    def answer(state: MessagesState):
        return {"messages": [bound_secondary.invoke(state["messages"])]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    # These nodes deliberately sit in front of the LLM graph.  Besides making
    # the integration tests deterministic, keeping the interrupt calls in the
    # graph (rather than in a tool) verifies the real LangGraph checkpoint and
    # resume path used by deployed agents.
    #
    # The parallel branch needs a mergeable "answers" channel; the extension is
    # additive, so the normal LLM/tool branch behaves exactly as MessagesState.
    class InterruptState(MessagesState):
        answers: Annotated[dict[str, Any], lambda left, right: {**left, **right}]

    def interrupt_single(state: MessagesState):
        resume_value = interrupt({"question": "single approval"})
        return {"messages": [AIMessage(content=json.dumps(resume_value))]}

    def wait_for_release(state: MessagesState):
        message = state["messages"][-1]
        token = UUID(str(message.content).split(":", 1)[1])
        release_file = Path("/tmp") / f"test-agent-release-{token.hex}"
        deadline = time.monotonic() + 90
        while not release_file.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Test did not release the running turn")
            time.sleep(0.05)
        release_file.unlink()
        return {"messages": [AIMessage(content="Released by test")]}

    def interrupt_left(state: InterruptState):
        resume_value = interrupt({"question": "left approval"})
        return {"answers": {"left": resume_value}}

    def interrupt_right(state: InterruptState):
        resume_value = interrupt({"question": "right approval"})
        return {"answers": {"right": resume_value}}

    def join_interrupts(state: InterruptState):
        return {"messages": [AIMessage(content=json.dumps(state["answers"]))]}

    def passthrough(state: MessagesState):
        return state

    def route_magic(state: MessagesState):
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                content = str(message.content)
                if content == "__interrupt_single__":
                    return "interrupt_single"
                if content == "__interrupt_parallel__":
                    return "interrupt_parallel"
                if content.startswith("__wait_for_release__:"):
                    return "wait_for_release"
                break
        return "rephrase"

    graph = StateGraph(InterruptState)
    # "route_magic" is the routing entry point; "interrupt_parallel" is a
    # do-nothing fan-out point whose two outgoing edges run both interrupt
    # nodes in the same superstep.
    graph.add_node("route_magic", passthrough)
    graph.add_node("rephrase", rephrase)
    graph.add_node("answer", answer)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("interrupt_single", interrupt_single)
    graph.add_node("wait_for_release", wait_for_release)
    graph.add_node("interrupt_parallel", passthrough)
    graph.add_node("interrupt_left", interrupt_left)
    graph.add_node("interrupt_right", interrupt_right)
    graph.add_node("join_interrupts", join_interrupts)
    graph.set_entry_point("route_magic")
    graph.add_conditional_edges("route_magic", route_magic)
    graph.add_edge("interrupt_single", END)
    graph.add_edge("wait_for_release", END)
    graph.add_edge("interrupt_parallel", "interrupt_left")
    graph.add_edge("interrupt_parallel", "interrupt_right")
    graph.add_edge("interrupt_left", "join_interrupts")
    graph.add_edge("interrupt_right", "join_interrupts")
    graph.add_edge("join_interrupts", END)
    graph.add_edge("rephrase", "answer")
    graph.add_conditional_edges("answer", should_continue)
    graph.add_edge("tools", "answer")
    return graph.compile(checkpointer=app.checkpointer())
