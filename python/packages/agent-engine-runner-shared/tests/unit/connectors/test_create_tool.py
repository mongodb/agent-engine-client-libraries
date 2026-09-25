"""Tests for the create-tool helper-container entrypoint."""

from __future__ import annotations

import json

import yaml

from agent_engine_runner_shared.connectors.create_tool import RESULT_PREFIX, main

from .test_generate import SPEC


def _run(capsys, argv):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _connector_dir(tmp_path):
    out = tmp_path / "connectors" / "jira"
    out.mkdir(parents=True)
    (out / "spec.yaml").write_text(SPEC, encoding="utf-8")
    return out


class TestCompileMode:
    def test_compiles_and_prints_sentinel_result(self, tmp_path, capsys, monkeypatch):
        out = _connector_dir(tmp_path)
        monkeypatch.chdir(tmp_path)

        code, stdout, _ = _run(
            capsys,
            [
                "--spec",
                "connectors/jira/spec.yaml",
                "--name",
                "jira",
                "--out-dir",
                "connectors/jira",
            ],
        )

        assert code == 0
        assert (out / "tool_defs.yaml").exists()
        assert (out / "tool.yaml").exists()
        result = json.loads(stdout.strip().removeprefix(RESULT_PREFIX))
        assert result["written"] == [
            "connectors/jira/tool_defs.yaml",
            "connectors/jira/tool.yaml",
        ]

    def test_scaffold_mode_with_allow_list(self, tmp_path, capsys, monkeypatch):
        out = _connector_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        main(
            [
                "--spec",
                "connectors/jira/spec.yaml",
                "--name",
                "jira",
                "--out-dir",
                "connectors/jira",
            ]
        )
        (out / "tool.yaml").unlink()

        code, _, _ = _run(
            capsys,
            [
                "--tool-defs",
                "tool_defs.yaml",
                "--base-url",
                "https://jira.corp/rest",
                "--allow",
                "jira_getForecast",
                "--out-dir",
                "connectors/jira",
            ],
        )

        assert code == 0
        authoring = yaml.safe_load((out / "tool.yaml").read_text(encoding="utf-8"))
        assert authoring["expose"] == {"allow": ["jira_getForecast"]}

    def test_regenerate_mode(self, tmp_path, capsys, monkeypatch):
        out = _connector_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        main(
            [
                "--spec",
                "connectors/jira/spec.yaml",
                "--name",
                "jira",
                "--out-dir",
                "connectors/jira",
            ]
        )
        (out / "spec.yaml").write_text(
            SPEC.replace("operationId: getForecast\n", "operationId: renamed\n"),
            encoding="utf-8",
        )

        monkeypatch.chdir(tmp_path)
        code, _, _ = _run(capsys, ["--config", "connectors/jira/tool.yaml"])

        assert code == 0
        catalog = (out / "tool_defs.yaml").read_text(encoding="utf-8")
        assert "jira_renamed" in catalog


class TestFailures:
    def test_generation_failure_exits_non_zero_with_error(self, tmp_path, capsys, monkeypatch):
        # A spec whose security requirement references an undefined scheme
        # must be refused with exit 1 and the scheme named in the error.
        monkeypatch.chdir(tmp_path)
        out2 = tmp_path / "connectors" / "ghost"
        out2.mkdir(parents=True)
        (out2 / "spec.yaml").write_text(
            SPEC.replace("servers:", "security:\n  - ghost: []\nservers:"), encoding="utf-8"
        )

        code, _, err = _run(
            capsys, ["--spec", str(out2 / "spec.yaml"), "--name", "jira", "--out-dir", str(out2)]
        )
        assert code == 1
        assert "error:" in err
        assert "ghost" in err

    def test_missing_name_is_a_generate_error(self, tmp_path, capsys, monkeypatch):
        out = _connector_dir(tmp_path)
        monkeypatch.chdir(tmp_path)

        code, _, err = _run(capsys, ["--spec", str(out / "spec.yaml"), "--out-dir", str(out)])

        assert code == 1
        assert "--name is required" in err

    def test_missing_base_url_in_scaffold_mode(self, tmp_path, capsys, monkeypatch):
        out = _connector_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        main(
            [
                "--spec",
                "connectors/jira/spec.yaml",
                "--name",
                "jira",
                "--out-dir",
                "connectors/jira",
            ]
        )
        (out / "tool.yaml").unlink()

        code, _, err = _run(capsys, ["--tool-defs", "tool_defs.yaml", "--out-dir", str(out)])

        assert code == 1
        assert "--base-url is required" in err

    def test_empty_allow_is_a_generate_error(self, tmp_path, capsys, monkeypatch):
        out = _connector_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        main(
            [
                "--spec",
                "connectors/jira/spec.yaml",
                "--name",
                "jira",
                "--out-dir",
                "connectors/jira",
            ]
        )
        (out / "tool.yaml").unlink()

        code, _, err = _run(
            capsys,
            [
                "--tool-defs",
                "tool_defs.yaml",
                "--base-url",
                "https://jira.corp/rest",
                "--allow",
                "",
                "--out-dir",
                "connectors/jira",
            ],
        )

        assert code == 1
        assert "--allow" in err

    def test_auth_without_env_is_a_generate_error(self, tmp_path, capsys, monkeypatch):
        _connector_dir(tmp_path)
        monkeypatch.chdir(tmp_path)

        code, _, err = _run(
            capsys,
            [
                "--spec",
                "spec.yaml",
                "--name",
                "jira",
                "--auth",
                "bearer",
                "--out-dir",
                "connectors/jira",
            ],
        )

        assert code == 1
        assert "--auth-env" in err

    def test_mutually_exclusive_modes_are_a_usage_refusal(self, capsys):
        """argparse usage errors are input validation: the helper maps them to
        the generation-refusal exit code instead of argparse's own status 2."""
        assert main(["--spec", "s.yaml", "--config", "t.yaml", "--out-dir", "."]) == 1

    def test_second_compile_is_refused_with_visible_error(self, tmp_path, capsys, monkeypatch):
        """Repeated compile refuses before either artifact changes — the CLI
        shows the error; intentional replacement belongs to --config."""
        monkeypatch.chdir(tmp_path)
        out = _connector_dir(tmp_path)
        assert (
            main(
                [
                    "--spec",
                    "connectors/jira/spec.yaml",
                    "--name",
                    "jira",
                    "--out-dir",
                    "connectors/jira",
                ]
            )
            == 0
        )
        catalog_before = (out / "tool_defs.yaml").read_bytes()
        authoring_before = (out / "tool.yaml").read_bytes()

        code, _, err = _run(
            capsys,
            [
                "--spec",
                "connectors/jira/spec.yaml",
                "--name",
                "jira",
                "--allow",
                "jira_getForecast",
                "--out-dir",
                "connectors/jira",
            ],
        )

        assert code == 1
        assert "refusing to overwrite" in err
        assert "--config" in err
        assert (out / "tool_defs.yaml").read_bytes() == catalog_before
        assert (out / "tool.yaml").read_bytes() == authoring_before

        # The refusal covers a plain repeat too: the catalog may hold hand
        # edits no tool can reconstruct.
        code, _, err = _run(
            capsys,
            [
                "--spec",
                "connectors/jira/spec.yaml",
                "--name",
                "jira",
                "--out-dir",
                "connectors/jira",
            ],
        )
        assert code == 1
        assert "refusing to overwrite" in err
        assert (out / "tool_defs.yaml").read_bytes() == catalog_before

    def test_fresh_compile_with_allow_writes_expose_allow(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = _connector_dir(tmp_path)

        code, _, _ = _run(
            capsys,
            [
                "--spec",
                "connectors/jira/spec.yaml",
                "--name",
                "jira",
                "--allow",
                "jira_getForecast",
                "--out-dir",
                "connectors/jira",
            ],
        )

        assert code == 0
        authoring = yaml.safe_load((out / "tool.yaml").read_text(encoding="utf-8"))
        assert authoring["expose"] == {"allow": ["jira_getForecast"]}

    def test_auth_flag_is_authority_over_the_spec_scheme(self, tmp_path, capsys, monkeypatch):
        """The official Jira DC document declares basic; a DC PAT is Bearer.
        --auth bearer must compile the catalog under the consumer's choice
        instead of rejecting the spec's scheme."""
        monkeypatch.chdir(tmp_path)
        out = _connector_dir(tmp_path)
        spec = out / "spec.yaml"
        spec.write_text(
            SPEC.replace(
                "servers:",
                """security:
  - basic: []
components:
  securitySchemes:
    basic: {type: http, scheme: basic}
servers:""",
            ),
            encoding="utf-8",
        )

        code, stdout, _ = _run(
            capsys,
            [
                "--spec",
                "connectors/jira/spec.yaml",
                "--name",
                "jira",
                "--auth",
                "bearer",
                "--auth-env",
                "JIRA_TOKEN",
                "--out-dir",
                "connectors/jira",
            ],
        )

        assert code == 0
        catalog = yaml.safe_load((out / "tool_defs.yaml").read_text(encoding="utf-8"))
        assert catalog["externalize"] == {"auth": {"type": "bearer"}}

    def test_auth_none_overrides_an_oauth_spec(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = _connector_dir(tmp_path)
        spec = out / "spec.yaml"
        spec.write_text(
            SPEC.replace(
                "servers:",
                """security:
  - oauth: []
components:
  securitySchemes:
    oauth: {type: oauth2, flows: {}}
servers:""",
            ),
            encoding="utf-8",
        )

        code, stdout, _ = _run(
            capsys,
            [
                "--spec",
                "connectors/jira/spec.yaml",
                "--name",
                "jira",
                "--auth",
                "none",
                "--out-dir",
                "connectors/jira",
            ],
        )

        assert code == 0
        catalog = yaml.safe_load((out / "tool_defs.yaml").read_text(encoding="utf-8"))
        # The explicit anonymous override is recorded as an empty auth
        # descriptor so the runtime binds no credential.
        assert catalog["externalize"] == {"auth": {}}

    def test_without_auth_flag_the_spec_scheme_governs(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = _connector_dir(tmp_path)
        spec = out / "spec.yaml"
        spec.write_text(
            SPEC.replace(
                "servers:",
                """security:
  - basic: []
components:
  securitySchemes:
    basic: {type: http, scheme: basic}
servers:""",
            ),
            encoding="utf-8",
        )

        code, _, _ = _run(
            capsys,
            [
                "--spec",
                "connectors/jira/spec.yaml",
                "--name",
                "jira",
                "--out-dir",
                "connectors/jira",
            ],
        )

        # Without --auth the document's own basic scheme resolves and is
        # recorded so the runtime can bind it with brokered credentials.
        assert code == 0
        catalog = yaml.safe_load((out / "tool_defs.yaml").read_text(encoding="utf-8"))
        assert catalog["externalize"] == {"auth": {"type": "basic"}}

    def test_out_dir_whose_name_starts_with_dots_is_contained(self, tmp_path, capsys, monkeypatch):
        """A directory named "..generated" is a contained out-dir, not a
        parent escape: relpath("…/workspace/../generated", "…/workspace") never
        begins with the segment "..", so the result line must succeed."""
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "..generated" / "jira"
        out.mkdir(parents=True)
        (out / "spec.yaml").write_text(SPEC, encoding="utf-8")

        code, stdout, _ = _run(
            capsys,
            [
                "--spec",
                str(out / "spec.yaml"),
                "--name",
                "jira",
                "--out-dir",
                str(out),
            ],
        )

        assert code == 0
        result = json.loads(stdout.strip().removeprefix(RESULT_PREFIX))
        assert all(not p.startswith("../") for p in result["written"])

    def test_scaffold_with_skip_unsupported_is_refused(self, tmp_path, capsys, monkeypatch):
        """Scaffolding references an existing catalog unchanged; a --skip-unsupported
        request would be a silent no-op, so the helper refuses it."""
        monkeypatch.chdir(tmp_path)
        _connector_dir(tmp_path)

        code, _, err = _run(
            capsys,
            [
                "--tool-defs",
                "tool_defs.yaml",
                "--base-url",
                "https://jira.corp/rest",
                "--skip-unsupported",
                "--out-dir",
                "connectors/jira",
            ],
        )

        assert code == 1
        assert "--skip-unsupported applies to --spec" in err

    def test_invalid_auth_choice_is_a_usage_refusal(self, tmp_path, capsys, monkeypatch):
        """argparse usage errors are input validation (exit 1), not the
        helper-failure exit 2 argparse would raise by itself."""
        monkeypatch.chdir(tmp_path)

        code, _, _ = _run(capsys, ["--spec", "spec.yaml", "--auth", "berer"])

        assert code == 1

    def test_operational_failure_exits_two(self, tmp_path, capsys, monkeypatch):
        """A non-validation exception (filesystem, environment) is helper-class:
        exit 2 with a sanitized single-line diagnostic, no traceback."""
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "connectors" / "jira"
        out.mkdir(parents=True)
        (out / "spec.yaml").write_text(SPEC, encoding="utf-8")
        monkeypatch.setattr(
            "agent_engine_runner_shared.connectors.create_tool.compile_catalog",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk on fire\nline two")),
        )

        code, _, err = _run(
            capsys, ["--spec", str(out / "spec.yaml"), "--name", "jira", "--out-dir", str(out)]
        )

        assert code == 2
        assert "disk on fire line two" in err
        assert "Traceback" not in err
