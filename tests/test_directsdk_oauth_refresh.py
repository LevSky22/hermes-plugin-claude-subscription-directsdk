"""Setup probes must not abandon Claude Code's OAuth refresh.

Every Claude Code command starts a refresh at init when the stored access token is inside its
five-minute window, and `auth status` exits before that refresh lands (anthropics/claude-code#95822).
The abandoned refresh leaves `<config>/.oauth_refresh.lock` behind, which Claude Code never reclaims
(anthropics/claude-code#95236), so every later native request fails with "another Claude Code process
is refreshing it or exited mid-refresh"; it can also spend the refresh token and force a re-login.
"""
import json
import os
import sys
import textwrap
import time

FAKE_CLI = textwrap.dedent('''
    import json, os, sys, time
    state = json.loads(os.environ["FAKE_STATE"])
    config = os.environ["CLAUDE_CONFIG_DIR"]
    lock = os.path.join(config, ".oauth_refresh.lock")
    def log(event):
        with open(os.environ["FAKE_LOG"], "a") as f:
            f.write(json.dumps({"event": event, "lock": os.path.exists(lock)}) + "\\n")
    if sys.argv[1:3] == ["auth", "status"]:
        log("auth-status")
        print(json.dumps(state["auth"])); sys.exit(0 if state["auth"]["loggedIn"] else 1)
    if "-p" in sys.argv and "/usage" in sys.argv:
        log("usage")
        mode = state.get("usage", "refresh")
        if mode == "hang":
            time.sleep(30)
        if mode == "fail":
            sys.exit(1)
        if not state["auth"]["loggedIn"]:
            sys.exit(1)
        path = os.path.join(config, ".credentials.json")
        if os.path.exists(path):
            creds = json.load(open(path))
            creds["claudeAiOauth"]["expiresAt"] = int(time.time() * 1000) + 8 * 3600 * 1000
            json.dump(creds, open(path, "w"))
        print("Current week (all models): 1% used"); sys.exit(0)
    sys.exit(f"unexpected argv {sys.argv}")
''')

PRO = {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"}


def _setup(tmp_path, *, expires_in=None, usage="refresh", lock=None, auth=PRO):
    """A fake CLI, its config dir and call log. ``expires_in`` (seconds) writes a file credential store;
    ``lock`` plants ``.oauth_refresh.lock`` as an empty directory aged that many seconds."""
    config = tmp_path / "config"
    config.mkdir()
    if expires_in is not None:
        creds = {"claudeAiOauth": {"accessToken": "a", "refreshToken": "r",
                                   "expiresAt": int((time.time() + expires_in) * 1000)}}
        (config / ".credentials.json").write_text(json.dumps(creds))
    if lock is not None:
        (config / ".oauth_refresh.lock").mkdir()
        stamp = time.time() - lock
        os.utime(config / ".oauth_refresh.lock", (stamp, stamp))
    cli = tmp_path / "claude.py"
    cli.write_text(FAKE_CLI)
    log = tmp_path / "calls.jsonl"
    env = {**os.environ, "PATH": os.defpath, "CLAUDE_CONFIG_DIR": str(config), "FAKE_LOG": str(log),
           "FAKE_STATE": json.dumps({"auth": auth, "usage": usage})}
    return [sys.executable, str(cli)], env, config, log


def _calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _expires_at(config):
    return json.loads((config / ".credentials.json").read_text())["claudeAiOauth"]["expiresAt"]


def test_expired_login_is_renewed_before_auth_status(profile, tmp_path):
    command, env, config, log = _setup(tmp_path, expires_in=-3600)
    status = profile.setup_status(command=command, env=env)
    assert status["logged_in"]
    assert [c["event"] for c in _calls(log)] == ["usage", "auth-status"]
    assert _expires_at(config) > time.time() * 1000


def test_logged_out_cli_still_reports_logged_out(profile, tmp_path):
    """The renewal must never mask a logout: the answer still comes from `auth status`."""
    command, env, _, log = _setup(tmp_path, auth={"loggedIn": False})
    status = profile.setup_status(command=command, env=env)
    assert not status["logged_in"] and "claude auth login" in status["detail"]
    assert [c["event"] for c in _calls(log)] == ["usage", "auth-status"]


def test_failed_renewal_still_reports_the_cli_login(profile, tmp_path):
    command, env, _, log = _setup(tmp_path, expires_in=-60, usage="fail")
    assert profile.setup_status(command=command, env=env)["logged_in"]
    assert [c["event"] for c in _calls(log)] == ["usage", "auth-status"]


def test_hung_renewal_is_bounded_by_the_probe_timeout(profile, tmp_path):
    command, env, _, log = _setup(tmp_path, expires_in=-60, usage="hang")
    started = time.monotonic()
    status = profile.setup_status(command=command, env=env, timeout=2)
    assert time.monotonic() - started < 10
    assert status["logged_in"]
    assert [c["event"] for c in _calls(log)] == ["usage", "auth-status"]


def test_orphaned_refresh_lock_is_reaped_before_the_cli_runs(profile, tmp_path):
    command, env, config, log = _setup(tmp_path, expires_in=3600, lock=120)
    profile.setup_status(command=command, env=env)
    assert _calls(log) == [{"event": "usage", "lock": False}, {"event": "auth-status", "lock": False}]
    assert not (config / ".oauth_refresh.lock").exists()


def test_a_live_refresh_lock_is_never_reaped(profile, tmp_path):
    """A refreshing Claude Code bumps the lock's mtime every 5 s; anything newer than its 60 s stale window is live."""
    command, env, config, log = _setup(tmp_path, expires_in=3600, lock=5)
    profile.setup_status(command=command, env=env)
    assert (config / ".oauth_refresh.lock").is_dir()


def test_a_lock_with_contents_is_never_reaped(profile, tmp_path):
    command, env, config, _ = _setup(tmp_path, expires_in=3600, lock=120)
    (config / ".oauth_refresh.lock" / "owner").write_text("pid")
    stamp = time.time() - 120
    os.utime(config / ".oauth_refresh.lock", (stamp, stamp))
    profile.setup_status(command=command, env=env)
    assert (config / ".oauth_refresh.lock" / "owner").exists()
