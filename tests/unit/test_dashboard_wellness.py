"""Dashboard: Garmin retry/backoff, per-day wellness cache and readiness verdict."""
import datetime
import json
import sys
from pathlib import Path

import pytest
from garminconnect import (
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard import app as dash  # noqa: E402

pytestmark = pytest.mark.unit

TODAY = datetime.date.today()


def day(offset: int) -> str:
    return (TODAY - datetime.timedelta(days=offset)).isoformat()


class FakeGarmin:
    """Minimal stand-in for garminconnect.Garmin's wellness endpoints."""

    def __init__(self, data=None, failing=()):
        self.data = data or {}  # date -> {"rhr", "bb", "sleep_s", "score", "hrv", "hrv_status"}
        self.failing = set(failing)  # (method, date) pairs that raise
        self.calls = []

    def _maybe_fail(self, method, d):
        self.calls.append((method, d))
        if (method, d) in self.failing:
            raise GarminConnectConnectionError("API client error (404): nope")

    def get_stats(self, d):
        self._maybe_fail("stats", d)
        v = self.data.get(d, {})
        return {"restingHeartRate": v.get("rhr"), "bodyBatteryMostRecentValue": v.get("bb")}

    def get_sleep_data(self, d):
        self._maybe_fail("sleep", d)
        v = self.data.get(d, {})
        if v.get("sleep_s") is None:
            return {"dailySleepDTO": {}}
        return {"dailySleepDTO": {
            "sleepTimeSeconds": v["sleep_s"],
            "sleepScores": {"overall": {"value": v.get("score")}},
        }}

    def get_hrv_data(self, d):
        self._maybe_fail("hrv", d)
        v = self.data.get(d, {})
        if v.get("hrv") is None:
            return None
        return {"hrvSummary": {"lastNightAvg": v["hrv"], "status": v.get("hrv_status", "BALANCED")}}


def full_day(rhr=50, sleep_h=7.5, hrv=60):
    return {"rhr": rhr, "bb": 70, "sleep_s": int(sleep_h * 3600), "score": 80, "hrv": hrv}


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "WELLNESS_CACHE_PATH", tmp_path / "wellness_cache.json")
    monkeypatch.setattr(dash, "COACH_DATA_DIR", tmp_path)
    monkeypatch.setattr(dash, "_wellness_cache", {})
    monkeypatch.setattr(dash.time, "sleep", lambda s: None)
    monkeypatch.setattr(dash, "_plan_for_date", lambda d: None)
    monkeypatch.setattr(dash, "_next_plan_after", lambda d: None)
    return tmp_path


# --------------------------------------------------------------------------- retry

class Flaky:
    def __init__(self, errors, result="ok"):
        self.errors = list(errors)
        self.result = result
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def test_retry_recovers_from_rate_limit():
    fn = Flaky([GarminConnectTooManyRequestsError("429")])
    assert dash._call_with_retry(fn) == "ok"
    assert fn.calls == 2


def test_retry_recovers_from_server_error():
    fn = Flaky([GarminConnectConnectionError("HTTP error: 503"), GarminConnectConnectionError("HTTP error: 502")])
    assert dash._call_with_retry(fn) == "ok"
    assert fn.calls == 3


def test_retry_gives_up_after_max_attempts():
    fn = Flaky([GarminConnectTooManyRequestsError("429")] * 5)
    with pytest.raises(GarminConnectTooManyRequestsError):
        dash._call_with_retry(fn)
    assert fn.calls == dash.RETRY_MAX_ATTEMPTS


@pytest.mark.parametrize("exc", [
    GarminConnectConnectionError("API client error (404): not found"),
    GarminConnectAuthenticationError("401"),
    ValueError("bug"),
])
def test_no_retry_on_client_auth_or_programming_errors(exc):
    fn = Flaky([exc])
    with pytest.raises(type(exc)):
        dash._call_with_retry(fn)
    assert fn.calls == 1


def test_backoff_honors_retry_after_header_and_caps_it():
    class Resp:
        headers = {"Retry-After": "3"}

    class HTTPErr(Exception):
        response = Resp()

    exc = GarminConnectTooManyRequestsError("429")
    exc.__cause__ = HTTPErr()
    assert dash._backoff_delay(1, exc) == 3.0

    Resp.headers = {"Retry-After": "999"}
    assert dash._backoff_delay(1, exc) == dash.RETRY_MAX_DELAY_S


def test_backoff_grows_exponentially_within_jitter():
    exc = GarminConnectTooManyRequestsError("429")
    for attempt, base in [(1, 0.5), (2, 1.0), (3, 2.0)]:
        delay = dash._backoff_delay(attempt, exc)
        assert base * 0.8 <= delay <= base * 1.2


def test_proxy_wraps_client_methods():
    class Client:
        n = 0

        def get_stats(self, d):
            Client.n += 1
            if Client.n == 1:
                raise GarminConnectTooManyRequestsError("429")
            return {"d": d}

    proxy = dash._RetryingGarmin(Client())
    assert proxy.get_stats("2026-01-01") == {"d": "2026-01-01"}
    assert Client.n == 2


# --------------------------------------------------------------------------- cache

def test_settled_past_day_is_cached_in_memory_and_on_disk(isolated_cache):
    client = FakeGarmin({day(3): full_day()})
    first = dash._wellness_days(client, TODAY - datetime.timedelta(days=3), TODAY - datetime.timedelta(days=3))
    assert first[0]["sleep_hours"] == 7.5 and first[0]["sleep_score"] == 80
    assert len(client.calls) == 3

    dash._wellness_days(client, TODAY - datetime.timedelta(days=3), TODAY - datetime.timedelta(days=3))
    assert len(client.calls) == 3  # served from cache

    on_disk = json.loads((isolated_cache / "wellness_cache.json").read_text(encoding="utf-8"))
    assert on_disk[day(3)]["hrv_avg"] == 60


def test_cache_survives_restart(isolated_cache):
    client = FakeGarmin({day(5): full_day()})
    dash._wellness_day(client, day(5))
    dash._save_wellness_cache_to_disk()
    dash._wellness_cache.clear()

    dash._load_wellness_cache_from_disk()
    client2 = FakeGarmin()
    assert dash._wellness_day(client2, day(5))["resting_hr"] == 50
    assert client2.calls == []


@pytest.mark.parametrize("offset", [0, 1])
def test_today_and_yesterday_always_refetched(offset):
    client = FakeGarmin({day(offset): full_day()})
    dash._wellness_day(client, day(offset))
    dash._wellness_day(client, day(offset))
    assert len(client.calls) == 6


def test_recent_day_without_sleep_is_not_cached_yet():
    client = FakeGarmin({day(3): {"rhr": 50}})  # watch not synced overnight data yet
    dash._wellness_day(client, day(3))
    assert day(3) not in dash._wellness_cache


def test_old_empty_day_is_cached():
    client = FakeGarmin({})  # e.g. watch not worn that day
    dash._wellness_day(client, day(dash.WELLNESS_SETTLED_DAYS))
    assert day(dash.WELLNESS_SETTLED_DAYS) in dash._wellness_cache


def test_failed_fetch_is_never_cached():
    client = FakeGarmin({day(10): full_day()}, failing={("hrv", day(10))})
    result = dash._wellness_day(client, day(10))
    assert result["errors"] == ["hrv"]
    assert result["sleep_hours"] == 7.5  # other sources still returned
    assert day(10) not in dash._wellness_cache


def test_old_entries_are_pruned_on_save(isolated_cache):
    ancient = (TODAY - datetime.timedelta(days=dash.WELLNESS_CACHE_MAX_AGE_DAYS + 1)).isoformat()
    dash._wellness_cache[ancient] = {"date": ancient}
    dash._wellness_cache[day(10)] = {"date": day(10)}
    dash._save_wellness_cache_to_disk()
    on_disk = json.loads((isolated_cache / "wellness_cache.json").read_text(encoding="utf-8"))
    assert list(on_disk) == [day(10)]


# ----------------------------------------------------------------------- readiness

def readiness_with(client, monkeypatch):
    monkeypatch.setattr(dash, "get_client", lambda: client)
    return dash.api_readiness()


def history(**today_overrides):
    data = {day(i): full_day() for i in range(1, 9)}
    data[day(0)] = {**full_day(), **today_overrides}
    return data


def test_readiness_is_pending_not_apto_when_sleep_missing(monkeypatch):
    r = readiness_with(FakeGarmin(history(sleep_s=None, hrv=None)), monkeypatch)
    assert r["verdict"] == "pendente"
    assert r["confidence"] == "parcial"
    assert any("ainda nao sincronizado" in x for x in r["reasons"])


def test_readiness_apto_with_high_confidence_when_all_data_present(monkeypatch):
    r = readiness_with(FakeGarmin(history()), monkeypatch)
    assert r["verdict"] == "apto"
    assert r["confidence"] == "alta"


def test_readiness_keeps_warning_flags_even_without_sleep(monkeypatch):
    r = readiness_with(FakeGarmin(history(sleep_s=None, rhr=60)), monkeypatch)  # RHR +10 vs baseline
    assert r["verdict"] == "cautela"
    assert r["confidence"] == "parcial"


def test_readiness_reports_sleep_fetch_failure_distinctly(monkeypatch):
    client = FakeGarmin(history(), failing={("sleep", day(0))})
    r = readiness_with(client, monkeypatch)
    assert r["verdict"] == "pendente"
    assert any("Falha ao obter o sono" in x for x in r["reasons"])


def test_endpoints_share_the_cache(monkeypatch):
    client = FakeGarmin(history())
    monkeypatch.setattr(dash, "get_client", lambda: client)
    dash.api_readiness()          # 9 days
    first = len(client.calls)
    dash.api_overtraining_risk(days=21)
    dash.api_recovery(days=14)
    # Only uncached days are fetched again: today+yesterday per endpoint, plus
    # the 12 older overtraining days not covered by readiness's window.
    assert len(client.calls) - first == 3 * (2 + 12) + 3 * 2
