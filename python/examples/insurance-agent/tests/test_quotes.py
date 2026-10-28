"""Insurance example quotes: loyalty needs a policy, good driver comes from memory.

The agent saves memory notes during a customer's first conversation, so notes
alone must not make a returning customer.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from insurance_agent import main

# A five-year-old vehicle on standard coverage, with no driver age, has no other
# price factors.
VEHICLE_YEAR = datetime.now().year - 5
BASE_PREMIUM = 120.0
LOYALTY = "Returning customer - 10% loyalty discount applied"
GOOD_DRIVER = "Good driver discount - additional 5% off"


class Memory:
    """Memory that holds a note from earlier in this conversation."""

    def __init__(self, context: str):
        self._context = context

    def search_episodes(self, **kwargs: Any) -> list[Any]:
        return [SimpleNamespace(title="New customer Priya registered")]

    def build_context(self, **kwargs: Any) -> Any:
        return SimpleNamespace(formatted_context=self._context)


def _quote(monkeypatch: pytest.MonkeyPatch, policies: list[dict], context: str = "") -> dict:
    app = SimpleNamespace(get_current_user_id=lambda: "user-1", memory=Memory(context))
    monkeypatch.setattr(main, "app", app)
    monkeypatch.setattr(main, "policy_store", SimpleNamespace(list_by_user_id=lambda _: policies))
    return json.loads(main.get_quote(VEHICLE_YEAR, "Toyota", "Camry", "standard"))


def _premium(quote: dict) -> float:
    return float(quote["monthly_premium"].lstrip("$"))


def test_new_customer_with_memory_notes_gets_no_loyalty_discount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote = _quote(monkeypatch, policies=[], context="Customer prefers email contact")
    assert "applied_discounts" not in quote
    assert _premium(quote) == BASE_PREMIUM
    assert "tip" in quote


def test_policyholder_gets_loyalty_discount(monkeypatch: pytest.MonkeyPatch) -> None:
    quote = _quote(monkeypatch, policies=[{"policy_number": "POL-1"}])
    assert quote["applied_discounts"] == [LOYALTY]
    assert _premium(quote) == pytest.approx(BASE_PREMIUM * 0.9)
    assert "tip" not in quote


def test_clean_driving_record_in_memory_gets_good_driver_discount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote = _quote(monkeypatch, policies=[], context="Customer has a clean driving record")
    assert quote["applied_discounts"] == [GOOD_DRIVER]
    assert _premium(quote) == pytest.approx(BASE_PREMIUM * 0.95)


def test_policyholder_with_clean_record_gets_both_discounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote = _quote(
        monkeypatch, policies=[{"policy_number": "POL-1"}], context="No accidents in 5 years"
    )
    assert quote["applied_discounts"] == [LOYALTY, GOOD_DRIVER]
    assert _premium(quote) == pytest.approx(BASE_PREMIUM * 0.85)
