"""Tests for the metadata secret reader and env applicator."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_engine_runner_shared.server.metadata import (
    applied_metadata_env,
    merge_metadata_env,
    read_metadata_secrets,
)


class TestReadMetadataSecrets:
    def test_reads_files_as_key_value_pairs(self, tmp_path: Path):
        (tmp_path / "STRIPE_KEY").write_text("sk_live_123")
        (tmp_path / "OPENAI_API_KEY").write_text("sk-openai-456")

        secrets = read_metadata_secrets(str(tmp_path))

        assert secrets == {"STRIPE_KEY": "sk_live_123", "OPENAI_API_KEY": "sk-openai-456"}

    def test_returns_empty_when_dir_missing(self, tmp_path: Path):
        assert read_metadata_secrets(str(tmp_path / "nonexistent")) == {}

    def test_skips_subdirectories(self, tmp_path: Path):
        (tmp_path / "SECRET").write_text("val")
        (tmp_path / "subdir").mkdir()

        secrets = read_metadata_secrets(str(tmp_path))

        assert secrets == {"SECRET": "val"}

    def test_skips_unreadable_files(self, tmp_path: Path):
        (tmp_path / "GOOD").write_text("ok")
        bad = tmp_path / "BAD"
        bad.write_text("nope")
        bad.chmod(0o000)

        secrets = read_metadata_secrets(str(tmp_path))

        assert "GOOD" in secrets
        bad.chmod(0o644)

    def test_returns_empty_for_empty_dir(self, tmp_path: Path):
        assert read_metadata_secrets(str(tmp_path)) == {}


class TestAppliedMetadataEnv:
    def test_injects_secrets_and_restores(self, tmp_path: Path):
        (tmp_path / "NEW_SECRET").write_text("secret_value")
        assert "NEW_SECRET" not in os.environ

        with applied_metadata_env(str(tmp_path)):
            assert os.environ["NEW_SECRET"] == "secret_value"

        assert "NEW_SECRET" not in os.environ

    def test_restores_overwritten_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("EXISTING_KEY", "original")
        (tmp_path / "EXISTING_KEY").write_text("overwritten")

        with applied_metadata_env(str(tmp_path)):
            assert os.environ["EXISTING_KEY"] == "overwritten"

        assert os.environ["EXISTING_KEY"] == "original"

    def test_noop_when_dir_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SENTINEL", "keep")

        with applied_metadata_env(str(tmp_path / "nonexistent")):
            pass

        assert os.environ["SENTINEL"] == "keep"

    def test_noop_when_dir_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SENTINEL", "keep")

        with applied_metadata_env(str(tmp_path)):
            pass

        assert os.environ["SENTINEL"] == "keep"

    def test_restores_on_exception(self, tmp_path: Path):
        (tmp_path / "TEMP_SECRET").write_text("temp")
        assert "TEMP_SECRET" not in os.environ

        with pytest.raises(RuntimeError, match="boom"):
            with applied_metadata_env(str(tmp_path)):
                assert os.environ["TEMP_SECRET"] == "temp"
                raise RuntimeError("boom")

        assert "TEMP_SECRET" not in os.environ

    def test_removes_keys_added_during_call(self, tmp_path: Path):
        """Keys added to os.environ by the tool itself are also cleaned up."""
        (tmp_path / "META_KEY").write_text("meta_val")

        with applied_metadata_env(str(tmp_path)):
            os.environ["TOOL_ADDED_KEY"] = "should_vanish"

        assert "META_KEY" not in os.environ
        assert "TOOL_ADDED_KEY" not in os.environ


class TestMergeMetadataEnv:
    def test_overwrites_idempotently_without_restore(self, tmp_path: Path):
        (tmp_path / "PERSIST_SECRET").write_text("keep-me")
        assert "PERSIST_SECRET" not in os.environ

        merge_metadata_env(str(tmp_path))
        merge_metadata_env(str(tmp_path))
        assert os.environ["PERSIST_SECRET"] == "keep-me"

        (tmp_path / "PERSIST_SECRET").write_text("rotated")
        merge_metadata_env(str(tmp_path))
        assert os.environ["PERSIST_SECRET"] == "rotated"
        del os.environ["PERSIST_SECRET"]


class TestOverlappingRegions:
    """Overlapping applied-env regions fail closed."""

    def test_secret_region_overlapping_secret_region_raises(self, tmp_path: Path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "SECRET_A").write_text("value_a")
        (dir_b / "SECRET_B").write_text("value_b")

        with applied_metadata_env(str(dir_a)):
            assert os.environ["SECRET_A"] == "value_a"
            with pytest.raises(RuntimeError, match="overlaps a concurrent secret-bearing region"):
                with applied_metadata_env(str(dir_b)):
                    pass
            # The rejected region must not have touched the live secrets.
            assert os.environ["SECRET_A"] == "value_a"
            assert "SECRET_B" not in os.environ

        assert "SECRET_A" not in os.environ
        assert "SECRET_B" not in os.environ

    def test_secretless_regions_may_overlap(self, tmp_path: Path):
        dir_a = tmp_path / "empty-a"
        dir_b = tmp_path / "empty-b"
        dir_a.mkdir()
        dir_b.mkdir()

        with applied_metadata_env(str(dir_a)):
            with applied_metadata_env(str(dir_b)):
                pass

    def test_overlapping_secretless_restore_does_not_resurrect_env_state(self, tmp_path: Path):
        """An inner region's stale snapshot must not re-add a key the outer
        region's restore already removed."""
        dir_a = tmp_path / "empty-a"
        dir_b = tmp_path / "empty-b"
        dir_a.mkdir()
        dir_b.mkdir()

        outer = applied_metadata_env(str(dir_a))
        outer.__enter__()
        os.environ["REQUEST_SCOPED"] = "a-value"
        inner = applied_metadata_env(str(dir_b))
        inner.__enter__()  # snapshots env *with* REQUEST_SCOPED
        outer.__exit__(None, None, None)  # removes REQUEST_SCOPED
        assert "REQUEST_SCOPED" not in os.environ
        inner.__exit__(None, None, None)
        assert "REQUEST_SCOPED" not in os.environ

    def test_nul_secret_value_does_not_wedge_later_regions(self, tmp_path: Path):
        """os.environ rejects NUL-containing values; the partial application is
        rolled back and the failed region is never counted, so later regions
        still work instead of every wrapper rejecting until restart."""
        dir_bad = tmp_path / "bad"
        dir_ok = tmp_path / "ok"
        dir_bad.mkdir()
        dir_ok.mkdir()
        (dir_bad / "A_GOOD_KEY").write_text("fine")
        (dir_bad / "NUL_KEY").write_text("bad\x00value")
        (dir_ok / "OK_SECRET").write_text("ok_val")

        with pytest.raises(ValueError):
            with applied_metadata_env(str(dir_bad)):
                pass
        assert "A_GOOD_KEY" not in os.environ
        assert "NUL_KEY" not in os.environ

        with applied_metadata_env(str(dir_ok)):
            assert os.environ["OK_SECRET"] == "ok_val"
        assert "OK_SECRET" not in os.environ

    def test_reinstates_keys_deleted_during_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("PRE_EXISTING", "keep-me")
        with applied_metadata_env(str(tmp_path)):
            del os.environ["PRE_EXISTING"]
        assert os.environ["PRE_EXISTING"] == "keep-me"

    def test_overlapping_restore_does_not_reassert_overwritten_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A (entered clean) overwrites a pre-existing key; B overlaps and
        snapshots A's request-scoped value. A's restore reverts to baseline —
        B's tainted restore must not write A's value back."""
        dir_a = tmp_path / "empty-a"
        dir_b = tmp_path / "empty-b"
        dir_a.mkdir()
        dir_b.mkdir()
        monkeypatch.setenv("BASELINE_KEY", "baseline")

        outer = applied_metadata_env(str(dir_a))
        outer.__enter__()
        os.environ["BASELINE_KEY"] = "a-value"
        inner = applied_metadata_env(str(dir_b))
        inner.__enter__()  # snapshots env with A's request-scoped value
        outer.__exit__(None, None, None)
        assert os.environ["BASELINE_KEY"] == "baseline"
        inner.__exit__(None, None, None)
        assert os.environ["BASELINE_KEY"] == "baseline"

    def test_cross_thread_secret_overlap_raises(self, tmp_path: Path):
        """A secret region on one thread must reject a secret region entered
        from another thread while it is live."""
        import threading

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "SECRET_A").write_text("value_a")
        (dir_b / "SECRET_B").write_text("value_b")

        entered = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        def hold_region() -> None:
            try:
                with applied_metadata_env(str(dir_a)):
                    entered.set()
                    release.wait(timeout=5)
            except BaseException as exc:  # pragma: no cover - defensive
                errors.append(exc)
                entered.set()

        thread = threading.Thread(target=hold_region)
        thread.start()
        try:
            assert entered.wait(timeout=5)
            assert not errors
            with pytest.raises(RuntimeError, match="overlaps a concurrent secret-bearing region"):
                with applied_metadata_env(str(dir_b)):
                    pass
            assert os.environ["SECRET_A"] == "value_a"
            assert "SECRET_B" not in os.environ
        finally:
            release.set()
            thread.join(timeout=5)

        assert "SECRET_A" not in os.environ
        assert "SECRET_B" not in os.environ

    def test_secret_region_overlapping_open_secret_stream_raises(self, tmp_path: Path):
        """Mirrors the TS stream-overlap test: ``_handle_invoke_llm_stream``
        yields SSE chunks *inside* ``with applied_metadata_env(...)``, so a
        region stays live across yields. A second secret region entered while
        the stream is mid-iteration must be rejected, and once the generator
        finishes a later region must succeed (the counters unwind)."""
        dir_stream = tmp_path / "stream"
        dir_other = tmp_path / "other"
        dir_stream.mkdir()
        dir_other.mkdir()
        (dir_stream / "STREAM_SECRET").write_text("stream_val")
        (dir_other / "OTHER_SECRET").write_text("other_val")

        def stream():
            with applied_metadata_env(str(dir_stream)):
                yield "chunk-1"
                yield "chunk-2"

        gen = stream()
        assert next(gen) == "chunk-1"  # region is now live across the yield
        assert os.environ["STREAM_SECRET"] == "stream_val"

        with pytest.raises(RuntimeError, match="overlaps a concurrent secret-bearing region"):
            with applied_metadata_env(str(dir_other)):
                pass
        # The rejected region must not have disturbed the stream's secrets.
        assert os.environ["STREAM_SECRET"] == "stream_val"
        assert "OTHER_SECRET" not in os.environ

        assert next(gen) == "chunk-2"
        with pytest.raises(StopIteration):
            next(gen)
        assert "STREAM_SECRET" not in os.environ

        # Counters unwound: a later secret region is admitted.
        with applied_metadata_env(str(dir_other)):
            assert os.environ["OTHER_SECRET"] == "other_val"
        assert "OTHER_SECRET" not in os.environ

    def test_failing_restore_does_not_wedge_later_regions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """If the restore raises, the counters must still unwind — otherwise
        every later /execute and /invoke_llm on the pod is rejected until
        restart."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "SECRET_A").write_text("value_a")
        (dir_b / "SECRET_B").write_text("value_b")

        real_pop = os.environ.pop

        def boom(key, *args):
            raise KeyError(key)

        with pytest.raises(KeyError):
            with applied_metadata_env(str(dir_a)):
                monkeypatch.setattr(os.environ, "pop", boom)
        monkeypatch.setattr(os.environ, "pop", real_pop)
        os.environ.pop("SECRET_A", None)

        with applied_metadata_env(str(dir_b)):
            assert os.environ["SECRET_B"] == "value_b"
        assert "SECRET_B" not in os.environ
