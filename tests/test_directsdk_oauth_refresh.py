"""Setup probes must not abandon Claude Code's OAuth refresh, and an orphaned refresh lock must not fail requests.

The ways this breaks, each pinned by a test below:

1. ``auth status`` on a login past its expiry starts the refresh at init and exits before it lands
   (anthropics/claude-code#95822). The lock is left behind and the refresh token may already be spent,
   which forces a re-login. ``setup_status`` runs on every ``/model`` switch and picker load.
2. Claude Code never reclaims that ``<config>/.oauth_refresh.lock`` (anthropics/claude-code#95236), so every
   later native request fails with "another Claude Code process is refreshing it or exited mid-refresh".
3. The lock is looked for in the wrong directory: the per-profile ``claude_config_dir`` setting (#34) wins
   over the environment in a multiplexed gateway, and on Windows ``~`` is %USERPROFILE%.
4. A live refresh (a lock younger than Claude Code's 60 s stale window) or a lock with contents is removed.
5. The renewal masks a logout, holds ``/model`` when it hangs, or changes the answer when it fails.
6. The renewal leaves traces: a session transcript per probe in the gateway's cwd, the user's MCP servers
   or hooks started on every ``/model`` switch.
"""
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

from conftest import PLUGIN_NAME
from test_directsdk import FAKE

FAKE_CLI = textwrap.dedent('''
    import json, os, sys, time
    state = json.loads(os.environ["FAKE_STATE"])
    config = os.environ["CLAUDE_CONFIG_DIR"]
    lock = os.path.join(config, ".oauth_refresh.lock")
    def log(event):
        with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, "lock": os.path.exists(lock), "config": config,
                                "argv": sys.argv[1:], "cwd": os.getcwd()}) + "\\n")
    if sys.argv[1:3] == ["auth", "status"]:
        log("auth-status")
        print(json.dumps(state["auth"])); sys.exit(0 if state["auth"]["loggedIn"] else 1)
    if sys.argv[1:3] == ["-p", "/usage"]:
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
            creds = json.load(open(path, encoding="utf-8-sig"))
            creds["claudeAiOauth"]["expiresAt"] = int(time.time() * 1000) + 8 * 3600 * 1000
            json.dump(creds, open(path, "w", encoding="utf-8"))
        print("Current week (all models): 1% used"); sys.exit(0)
    sys.exit(f"unexpected argv {sys.argv}")
''')

PRO = {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"}


def _config(path, *, expires_in=None, lock=None):
    """A Claude config dir. ``expires_in`` (seconds) writes a file credential store; ``lock`` plants
    ``.oauth_refresh.lock`` as an empty directory aged that many seconds."""
    path.mkdir()
    if expires_in is not None:
        creds = {"claudeAiOauth": {"accessToken": "a", "refreshToken": "r",
                                   "expiresAt": int((time.time() + expires_in) * 1000)}}
        (path / ".credentials.json").write_text(json.dumps(creds), encoding="utf-8")
    if lock is not None:
        _age((path / ".oauth_refresh.lock"), lock, create=True)
    return path


def _age(path, seconds, create=False):
    if create:
        path.mkdir()
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def _setup(tmp_path, *, expires_in=None, usage="refresh", lock=None, auth=PRO):
    """A fake CLI, its config dir and call log."""
    config = _config(tmp_path / "config", expires_in=expires_in, lock=lock)
    cli = tmp_path / "claude.py"
    cli.write_text(FAKE_CLI, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    env = {**os.environ, "PATH": os.defpath, "CLAUDE_CONFIG_DIR": str(config), "FAKE_LOG": str(log),
           "FAKE_STATE": json.dumps({"auth": auth, "usage": usage})}
    return [sys.executable, str(cli)], env, config, log


def _calls(log):
    return [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()] if log.exists() else []


def _events(log):
    return [(c["event"], c["lock"]) for c in _calls(log)]


def _expires_at(config):
    return json.loads((config / ".credentials.json").read_text(encoding="utf-8-sig"))["claudeAiOauth"]["expiresAt"]


def _set_config_dir(value):
    from hermes_cli.config import get_config_path
    get_config_path().write_text(
        "plugins:\n  enabled: []\n  entries:\n    %s:\n      settings:\n        claude_config_dir: %s\n"
        % (PLUGIN_NAME, json.dumps(value)), encoding="utf-8")


def test_expired_login_is_renewed_before_auth_status(profile, tmp_path):
    command, env, config, log = _setup(tmp_path, expires_in=-3600)
    status = profile.setup_status(command=command, env=env)
    assert status["logged_in"]
    assert [c["event"] for c in _calls(log)] == ["usage", "auth-status"]
    assert _expires_at(config) > time.time() * 1000


def test_model_discovery_renews_once_before_the_handshake(profile, tmp_path):
    """discover_models gates on setup_status, so the handshake starts on a renewed login."""
    command, env, _, log = _setup(tmp_path, expires_in=-3600)
    profile.discover_models(command=command, env=env)  # the fake has no handshake: the picker falls back
    assert [c["event"] for c in _calls(log)] == ["usage", "auth-status"]


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


def test_the_renewal_leaves_no_session_and_starts_nothing_of_the_users(profile, tmp_path):
    """No transcript per `/model` switch in the gateway's cwd, no user settings, hooks or MCP servers. Slash
    commands stay on (`/usage` is one), and the prompt comes first: `--mcp-config` takes several values."""
    command, env, _, log = _setup(tmp_path, expires_in=3600)
    profile.setup_status(command=command, env=env)
    usage = _calls(log)[0]
    argv = usage["argv"]
    assert argv[:2] == ["-p", "/usage"]
    assert "--no-session-persistence" in argv and "--strict-mcp-config" in argv
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert json.loads(argv[argv.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert "--disable-slash-commands" not in argv
    assert Path(usage["cwd"]).resolve() != Path.cwd().resolve() and not Path(usage["cwd"]).exists()


def test_orphaned_refresh_lock_is_reaped_before_the_cli_runs(profile, tmp_path):
    command, env, config, log = _setup(tmp_path, expires_in=3600, lock=120)
    profile.setup_status(command=command, env=env)
    assert _events(log) == [("usage", False), ("auth-status", False)]
    assert not (config / ".oauth_refresh.lock").exists()


def test_a_live_refresh_lock_is_never_reaped(profile, tmp_path):
    """A refreshing Claude Code bumps the lock's mtime every 5 s; anything newer than its 60 s stale window is live."""
    command, env, config, _ = _setup(tmp_path, expires_in=3600, lock=5)
    profile.setup_status(command=command, env=env)
    assert (config / ".oauth_refresh.lock").is_dir()


def test_a_lock_with_contents_is_never_reaped(profile, tmp_path):
    command, env, config, _ = _setup(tmp_path, expires_in=3600, lock=120)
    (config / ".oauth_refresh.lock" / "owner").write_text("pid", encoding="utf-8")
    _age(config / ".oauth_refresh.lock", 120)
    profile.setup_status(command=command, env=env)
    assert (config / ".oauth_refresh.lock" / "owner").exists()


def test_the_profile_login_directory_is_renewed_and_reaped(profile, tmp_path):
    """#34: a profile's claude_config_dir wins over the environment, so that is the login to renew and the lock
    to reap. The environment's own directory is another login, and stays untouched."""
    command, env, inherited, log = _setup(tmp_path, expires_in=3600, lock=120)
    selected = _config(tmp_path / "profile-login", expires_in=-3600, lock=120)
    _set_config_dir(str(selected))
    assert profile.setup_status(command=command, env=env)["logged_in"]
    assert {c["config"] for c in _calls(log)} == {str(selected)}
    assert _events(log) == [("usage", False), ("auth-status", False)]
    assert _expires_at(selected) > time.time() * 1000
    assert (inherited / ".oauth_refresh.lock").is_dir()


def test_a_native_request_reaps_the_lock_of_the_profile_login(profile, tmp_path):
    import directsdk
    selected = _config(tmp_path / "profile-login", lock=120)
    _set_config_dir(str(selected))
    capture = tmp_path / "lock.jsonl"
    probe = ("open(os.environ['LOCK_CAPTURE'], 'a').write(json.dumps(os.path.exists(os.path.join("
             "os.environ['CLAUDE_CONFIG_DIR'], '.oauth_refresh.lock'))) + '\\n')\n")
    script = tmp_path / "native.py"
    script.write_text(FAKE.replace("rows=[]", probe + "rows=[]"), encoding="utf-8")
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "LOCK_CAPTURE": str(capture)}
    client = directsdk.Client(command=[sys.executable, str(script)], env=env)
    try:
        reply = client.create(model="sonnet", messages=[{"role": "user", "content": "go"}],
                              tools=[{"type": "function", "function": {"name": "probe", "description": "TAIL",
                                                                       "parameters": {"type": "object", "properties": {}}}}])
    finally:
        client.close()
    assert reply.choices[0].message.content == "hello\n"
    assert capture.read_text(encoding="utf-8-sig").split() == ["false"]
    assert not (selected / ".oauth_refresh.lock").exists()


def test_usage_reaps_the_lock_of_the_profile_login_before_asking(profile, tmp_path, monkeypatch):
    import directsdk_setup
    selected = _config(tmp_path / "profile-login", lock=120)
    _set_config_dir(str(selected))
    seen = []

    def run(argv, **kwargs):
        seen.append((kwargs["env"]["CLAUDE_CONFIG_DIR"], (selected / ".oauth_refresh.lock").exists()))
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(directsdk_setup.subprocess, "run", run)
    directsdk_setup.fetch_usage(command=[sys.executable], env={"PATH": os.defpath, "HOME": str(tmp_path)})
    assert seen == [(str(selected), False)]


def test_config_dir_follows_the_cli_home_rules():
    """Read from the child env, after the profile's directory was applied. Without one, Claude Code resolves ~ with
    Node's os.homedir(): %USERPROFILE% on Windows even when a Git Bash or MSYS shell sets HOME elsewhere."""
    from directsdk_setup import _config_dir

    env = {"HOME": "/msys/home/user", "USERPROFILE": "C:/Users/user"}
    assert _config_dir(env, windows=True) == Path("C:/Users/user") / ".claude"
    assert _config_dir(env, windows=False) == Path("/msys/home/user") / ".claude"
    assert _config_dir({**env, "CLAUDE_CONFIG_DIR": "/cfg"}, windows=True) == Path("/cfg")
