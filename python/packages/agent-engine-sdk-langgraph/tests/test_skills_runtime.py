"""Runtime skill discovery through the SDK deep-agent factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_engine_sdk_langgraph.deep_agent import (
    _load_skill_paths,
    create_agent_engine_deep_agent,
)


class RecordingChatModel(BaseChatModel):
    """A tool-bindable fake that retains the prompt sent by deepagents."""

    prompts: list[list[BaseMessage]] = []
    spawn_subagent: bool = False
    _task_spawned: bool = False

    def __init__(self, *, spawn_subagent: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.prompts = []
        self.spawn_subagent = spawn_subagent
        self._task_spawned = False

    @property
    def _llm_type(self) -> str:
        return "recording-chat-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> RecordingChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(messages)
        if self.spawn_subagent and not self._task_spawned:
            self._task_spawned = True
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "call_reviewer",
                                    "name": "task",
                                    "args": {
                                        "subagent_type": "reviewer",
                                        "description": "Review the change.",
                                    },
                                }
                            ],
                        )
                    )
                ]
            )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="done"))]
        )


def _write_skill(parent: Path, name: str) -> None:
    skill_dir = parent / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} instructions\n---\n\n# {name}\n",
        encoding="utf-8",
    )


def _prompt_text(model: RecordingChatModel) -> str:
    return "\n".join(str(message.content) for message in model.prompts[0])


def test_load_skill_paths_preserves_absolute_path_with_base_dir(tmp_path: Path) -> None:
    absolute_path = str(tmp_path / "skills")

    assert _load_skill_paths([absolute_path], base_dir=tmp_path / "agent") == [
        absolute_path
    ]


def test_load_skill_paths_preserves_relative_path_without_base_dir() -> None:
    assert _load_skill_paths(["skills"], base_dir=None) == ["skills"]


def test_factory_discovers_skills_from_relative_parent_directory(
    tmp_path: Path,
) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "security-review")
    model = RecordingChatModel()

    agent = create_agent_engine_deep_agent(
        secure_llm=model,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=False),
        skills=["skills"],
        skills_base_dir=tmp_path,
    )
    agent.invoke({"messages": [HumanMessage(content="Review this change")]})

    prompt = _prompt_text(model)
    assert "## Skills System" in prompt
    assert "security-review instructions" in prompt
    assert str(skills_dir / "security-review" / "SKILL.md") in prompt


def test_factory_resolves_relative_subagent_skill_parent_directories(
    tmp_path: Path,
) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "specialist")
    model = RecordingChatModel(spawn_subagent=True)

    agent = create_agent_engine_deep_agent(
        secure_llm=model,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=False),
        subagents=[
            {
                "name": "reviewer",
                "description": "Reviews changes",
                "system_prompt": "Review the change.",
                "model": model,
                "skills": ["skills"],
            }
        ],
        skills_base_dir=tmp_path,
    )

    agent.invoke({"messages": [HumanMessage(content="Review this change")]})

    assert any(
        "specialist instructions"
        in "\n".join(str(message.content) for message in prompt)
        for prompt in model.prompts
    )
    assert any(
        str(skills_dir / "specialist" / "SKILL.md")
        in "\n".join(str(message.content) for message in prompt)
        for prompt in model.prompts
    )
