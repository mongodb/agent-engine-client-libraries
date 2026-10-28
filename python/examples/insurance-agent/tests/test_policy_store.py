"""The insurance example's stores recover once MongoDB becomes reachable.

An agent can start before it can reach its database, for example before Atlas
egress is ready. The stores must not stay broken for the life of the process:
a call made while the database is unreachable fails, and a later call succeeds.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from types import ModuleType

import pytest
from pymongo.errors import ConfigurationError, ServerSelectionTimeoutError

from insurance_agent import policy_store


class FakeCollection:
    def __init__(self, client: FakeClient):
        self._client = client

    def create_index(self, keys: str, unique: bool = False) -> None:
        if self._client.index_error:
            raise self._client.index_error("index creation failed")

    def find(self, query: dict, projection: dict) -> list[dict]:
        return [{"user_id": query["user_id"], "policy_number": "POL-1"}]


class FakeClient:
    def __init__(self, index_error: type[BaseException] | None = None):
        self.index_error = index_error
        self.closed = False

    def __getitem__(self, name: str) -> dict[str, FakeCollection]:
        collection = FakeCollection(self)
        return {"insurance_policies": collection, "insurance_claims": collection}

    def close(self) -> None:
        self.closed = True


class Database:
    """Stands in for ``MongoClient``: unreachable at first, reachable later.

    The first connection fails resolving the ``mongodb+srv`` address, the second
    connects but cannot reach a server to create indexes, and later ones succeed.
    """

    def __init__(self) -> None:
        self.attempts = 0
        self.kwargs: dict[str, object] = {}
        self.clients: list[FakeClient] = []

    def __call__(self, uri: str, **kwargs: object) -> FakeClient:
        self.attempts += 1
        self.kwargs = kwargs
        if self.attempts == 1:
            raise ConfigurationError("The DNS query name does not exist")
        client = FakeClient(ServerSelectionTimeoutError if self.attempts == 2 else None)
        self.clients.append(client)
        return client


def test_store_recovers_after_database_becomes_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database()
    monkeypatch.setattr(policy_store, "MongoClient", database)

    store = policy_store.create_policy_store("mongodb+srv://cluster.example.net")
    assert store is not None, "creating the store must not need the database"

    with pytest.raises(ConfigurationError):
        store.list_by_user_id("user-1")
    with pytest.raises(ServerSelectionTimeoutError):
        store.list_by_user_id("user-1")
    assert database.clients[0].closed, "a client that failed to set up is closed"

    assert store.list_by_user_id("user-1") == [{"user_id": "user-1", "policy_number": "POL-1"}]
    store.list_by_user_id("user-1")
    assert len(database.clients) == 2, "a working connection is reused"
    # The mongodb+srv DNS lookup and server selection fail fast, and a server that
    # stops answering cannot hold a call forever.
    assert database.kwargs == {
        "connectTimeoutMS": 2000,
        "serverSelectionTimeoutMS": 2000,
        "socketTimeoutMS": 10000,
    }


def test_claim_store_is_created_without_the_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_store, "MongoClient", Database())
    assert policy_store.create_claim_store("mongodb+srv://cluster.example.net") is not None


def test_no_store_without_a_uri() -> None:
    assert policy_store.create_policy_store("") is None
    assert policy_store.create_claim_store("") is None


class Interrupted(BaseException):
    """Ends a call the way ``KeyboardInterrupt`` or ``SystemExit`` would."""


class SlowDatabase(Database):
    """The first connection hangs until released, then raises ``first_error`` if set."""

    def __init__(self, first_error: type[BaseException] | None = None) -> None:
        super().__init__()
        self.first_error = first_error
        self.connecting = threading.Event()
        self.release = threading.Event()

    def __call__(self, uri: str, **kwargs: object) -> FakeClient:
        self.attempts += 1
        if self.attempts == 1:
            self.connecting.set()
            self.release.wait(5)
            if self.first_error:
                raise self.first_error("first connection failed")
        client = FakeClient()
        self.clients.append(client)
        return client


class JoinedFuture(Future):
    """A connection attempt that signals when a second call waits on it."""

    joined = threading.Event()

    def result(self, timeout: float | None = None) -> object:
        JoinedFuture.joined.set()
        return super().result(timeout)


def _overlapping_calls(module: ModuleType, store, monkeypatch, database: SlowDatabase) -> list:
    """Run two calls that overlap one connection attempt; return their outcomes."""
    JoinedFuture.joined = threading.Event()
    monkeypatch.setattr(module, "Future", JoinedFuture)
    outcomes: list = [None, None]

    def call(index: int) -> None:
        try:
            outcomes[index] = store.list_by_user_id("user-1")
        except BaseException as error:
            outcomes[index] = error

    # Daemon threads, so a call left waiting fails the test instead of hanging it.
    first = threading.Thread(target=call, args=(0,), daemon=True)
    first.start()
    assert database.connecting.wait(5)
    second = threading.Thread(target=call, args=(1,), daemon=True)
    second.start()
    assert JoinedFuture.joined.wait(5), "the second call joins the attempt in flight"
    database.release.set()
    first.join(5)
    second.join(5)
    return outcomes


def test_overlapping_calls_share_a_successful_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    database = SlowDatabase()
    monkeypatch.setattr(policy_store, "MongoClient", database)
    store = policy_store.create_policy_store("mongodb+srv://cluster.example.net")

    outcomes = _overlapping_calls(policy_store, store, monkeypatch, database)
    assert outcomes == [[{"user_id": "user-1", "policy_number": "POL-1"}]] * 2
    assert database.attempts == 1


def test_overlapping_calls_share_a_failed_attempt_then_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = SlowDatabase(ServerSelectionTimeoutError)
    monkeypatch.setattr(policy_store, "MongoClient", database)
    store = policy_store.create_policy_store("mongodb+srv://cluster.example.net")

    outcomes = _overlapping_calls(policy_store, store, monkeypatch, database)
    assert [type(outcome) for outcome in outcomes] == [ServerSelectionTimeoutError] * 2
    assert database.attempts == 1, "a call sharing a failed attempt does not start another"

    assert store.list_by_user_id("user-1") == [{"user_id": "user-1", "policy_number": "POL-1"}]
    store.list_by_user_id("user-1")
    assert database.attempts == 2, "the next call reconnects, and later calls reuse it"


def test_overlapping_call_is_released_when_the_attempt_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = SlowDatabase(Interrupted)
    monkeypatch.setattr(policy_store, "MongoClient", database)
    store = policy_store.create_policy_store("mongodb+srv://cluster.example.net")

    outcomes = _overlapping_calls(policy_store, store, monkeypatch, database)
    assert [type(outcome) for outcome in outcomes] == [Interrupted] * 2

    assert store.list_by_user_id("user-1") == [{"user_id": "user-1", "policy_number": "POL-1"}]


class InterruptsIndexing(Database):
    """The first connection is interrupted while creating indexes; later ones succeed."""

    def __call__(self, uri: str, **kwargs: object) -> FakeClient:
        self.attempts += 1
        client = FakeClient(Interrupted if self.attempts == 1 else None)
        self.clients.append(client)
        return client


def test_client_is_closed_when_index_creation_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = InterruptsIndexing()
    monkeypatch.setattr(policy_store, "MongoClient", database)
    store = policy_store.create_policy_store("mongodb+srv://cluster.example.net")

    with pytest.raises(Interrupted):
        store.list_by_user_id("user-1")
    assert database.clients[0].closed

    assert store.list_by_user_id("user-1") == [{"user_id": "user-1", "policy_number": "POL-1"}]


class ConnectsWhileWaiting:
    """A lock taken just after another call finished connecting."""

    def __init__(self, lazy, collection: FakeCollection):
        self._lazy = lazy
        self._collection = collection

    def __enter__(self) -> None:
        self._lazy._collection = self._collection

    def __exit__(self, *exc: object) -> None:
        pass


def test_call_uses_a_connection_made_while_it_waited_for_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database()
    monkeypatch.setattr(policy_store, "MongoClient", database)
    store = policy_store.create_policy_store("mongodb+srv://cluster.example.net")
    store._lazy._lock = ConnectsWhileWaiting(store._lazy, FakeCollection(FakeClient()))

    assert store.list_by_user_id("user-1") == [{"user_id": "user-1", "policy_number": "POL-1"}]
    assert database.attempts == 0, "a connection made meanwhile is used, not repeated"
