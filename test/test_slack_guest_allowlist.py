"""Tests for the Slack guest allowlist: admission, routing and capability limits.

An allow-listed guest is a non-owner who may reach ONE thing — the inbound message
gate in a tracked channel. These tests pin that, and pin the boundary around it:
every owner control still refuses them, the turn resolves a memory store that is
not the owner's or does not run, and its tools are deny-by-default.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro_crew.agent_files import SLACK_GUEST_AGENT_FILENAME, SLACK_GUEST_AGENT_NAME
from kiro_crew.config.loader import (
    ACTIVATION_ALWAYS,
    ACTIVATION_MENTION,
    ACTIVATION_OFF,
    ACTIVATION_REVIEW,
    ChannelConfig,
    KiroCrewConfig,
    MessagingConfig,
)
from kiro_crew.memory_stores import DEFAULT_MEMORY_STORE
from kiro_crew.messaging import privacy_mode
from kiro_crew.messaging.link import canonical_key
from kiro_crew.slack.events import (
    _GUEST_DENIAL_TEXT,
    SeenCache,
    _denial_ephemeral,
    _resolve_approval_mode,
    _route_message,
)
from kiro_crew.slack.handler import (
    APPROVAL_AUTO,
    APPROVAL_INTERACTIVE,
    guest_member_configured,
    guest_session_key,
    is_allowed_user,
    is_guest_session_key,
    is_guest_user,
    maybe_apply_privacy_modifiers,
    maybe_handle_keyword_command,
    resolve_guest_agent,
    set_allowed_users,
    set_owner_id,
    set_tracking_channels,
)
from kiro_crew.slack.tool_gate import (
    GUEST_SAFE_TOOLS,
    build_guest_hooks,
    is_guest_safe_tool,
)

OWNER = "U0OWNER01"
GUEST = "U0GUEST01"
STRANGER = "U0STRANGE"
TRACKED = "C0TRACKED"
UNTRACKED = "C0UNTRACK"
GUEST_MEMBER = "guest-member"


@pytest.fixture(autouse=True)
def _identities():
    """Owner set, guest allow-listed, one tracked channel."""
    set_owner_id(OWNER)
    set_allowed_users({OWNER, GUEST})
    set_tracking_channels({TRACKED})
    yield
    set_owner_id("")
    set_allowed_users(set())
    set_tracking_channels(set())


#: Distinguishes "caller said nothing" from "caller passed None". A test models a
#: config that cannot be read by passing None explicitly, so the default cannot be
#: None itself.
_UNSET = object()


def _cfg(activation: str = ACTIVATION_MENTION, *, guest_agent: str = GUEST_MEMBER, **kw):
    """Config with TRACKED at *activation* and a guest member configured."""
    cfg = KiroCrewConfig(
        slack_channels={
            TRACKED: ChannelConfig(activation=activation),
            UNTRACKED: ChannelConfig(activation=activation),
        },
        messaging=MessagingConfig(use_transport=False),
        **kw,
    )
    cfg.slack.guest_agent = guest_agent
    if guest_agent:
        # Both halves item 2's predicate demands: a store that is not the owner's,
        # and a binding to the generated guest spec. A member missing either is
        # refused admission, so a fixture setting only one would test the refusal.
        cfg.agents[guest_agent] = MagicMock(
            memory_store="guest-store", kiro_agent=SLACK_GUEST_AGENT_NAME, member_id=""
        )
    return cfg


def _real_index(tmp_path):
    """A REAL ``SessionMap``, isolated to *tmp_path*.

    Real, because the guest-claim rule lives in the index now: a MagicMock returns
    whatever it was handed and would pass whatever the filter did.

    Isolated, because ``set_slack_link`` takes a ``setdefault`` branch for a
    self-derived key -- a claim left behind by an earlier test would NOT be
    displaced, so a shared map makes these tests order-dependent in the exact
    behaviour they exist to check.
    """
    from kiro_crew.session_map import SESSION_MAP_FILENAME, SessionMap

    smap = SessionMap()
    smap._path = tmp_path / SESSION_MAP_FILENAME
    smap._data = {}
    smap._thread_to_session = {}
    return smap


def _make_orch(cfg=None) -> MagicMock:
    orch = MagicMock()
    orch._cfg = cfg if cfg is not None else _cfg()
    orch.channel_history = MagicMock()
    orch.slack = MagicMock()
    orch.slack.post_ephemeral = AsyncMock()
    orch.sessions = AsyncMock()
    orch.sessions.enqueue = MagicMock(return_value=False)
    orch.sessions.is_busy = MagicMock(return_value=False)
    orch.sessions.is_cancelled = MagicMock(return_value=False)
    orch.sessions.dequeue = MagicMock(return_value=None)
    orch.sessions.clear_queue = MagicMock()
    orch.sessions.has_session = MagicMock(return_value=False)
    orch.sessions.get_session_for_thread = MagicMock(return_value=None)
    orch.ctx_builder = None
    orch.cron_svc = None
    orch.conv_log = None
    orch.consolidator = None
    orch.subagent_mgr = None
    orch.task_runner = None
    orch._handler_tasks = set()
    orch._session_tasks = {}
    orch._pending_queue = {}
    orch._approval_mode = None
    return orch


async def _route(
    orch,
    *,
    channel=TRACKED,
    user=GUEST,
    is_mention=True,
    ts="10.0",
    thread=None,
    text="hi",
    loaded_cfg=_UNSET,
):
    """Drive one inbound message and return the handle_message mock.

    ``loaded_cfg`` is what ``KiroCrewConfig.load()`` returns during this turn.
    Admission resolves the config it admits on rather than reading ``orch._cfg``
    for the ``agents`` section, because that snapshot is not reloaded for sections
    outside ``_SLACK_OWNED_FIELDS`` -- so a test that patched only ``orch._cfg``
    would be describing an object admission does not consult. It defaults to
    ``orch._cfg``, i.e. disk AGREES with the snapshot, which is the ordinary case;
    pass a different object to model an operator editing the member mid-turn.

    ``text`` is the message body. It matters for the commands ``_route_message``
    intercepts itself rather than handing to ``handle_message`` -- ``!stop`` is
    the one this file drives -- because those never reach the returned mock.
    """
    seen = SeenCache()
    event = {"user": user, "channel": channel, "text": text, "ts": ts, "team": "TTEST"}
    if thread:
        event["thread_ts"] = thread
    _loaded = orch._cfg if loaded_cfg is _UNSET else loaded_cfg
    with (
        patch("kiro_crew.slack.events.handle_message", new_callable=AsyncMock) as hm,
        patch.object(KiroCrewConfig, "load", staticmethod(lambda *a, **k: _loaded)),
    ):
        await _route_message(orch, event, seen, is_mention=is_mention)
        await asyncio.sleep(0)
        await asyncio.gather(*list(orch._handler_tasks), return_exceptions=True)
        return hm


# ───────────────────────── the two predicates are disjoint ─────────────────────


class TestPredicatesAreDisjoint:
    def test_guest_is_not_an_allowed_user(self):
        assert is_guest_user(GUEST) is True
        assert is_allowed_user(GUEST) is False

    def test_owner_is_not_a_guest(self):
        assert is_allowed_user(OWNER) is True
        assert is_guest_user(OWNER) is False

    def test_stranger_is_neither(self):
        assert is_guest_user(STRANGER) is False
        assert is_allowed_user(STRANGER) is False


# ───────────────────────────── inbound admission ───────────────────────────────


class TestGuestAdmission:
    @pytest.mark.asyncio
    async def test_guest_mention_in_tracked_channel_is_answered(self):
        orch = _make_orch()
        hm = await _route(orch)
        hm.assert_called_once()
        assert hm.call_args.kwargs["guest_user"] == GUEST

    @pytest.mark.asyncio
    @pytest.mark.parametrize("activation", [ACTIVATION_MENTION, ACTIVATION_ALWAYS])
    async def test_guest_is_dispatched_with_its_guest_flag(self, activation):
        """Admission passes the guest id on; WHICH AGENT runs is a seam test.

        ``channel_agent`` is a ``config.agents`` key that the layer below re-derives,
        and the provider never sees it, so an assertion on it here says nothing about
        the agent that runs. The agent that crosses into the session layer is pinned
        in ``TestGuestAgentSeam``; what this pins is that the guest id travels.
        """
        orch = _make_orch(_cfg(activation))
        hm = await _route(orch)
        hm.assert_called_once()
        assert hm.call_args.kwargs["guest_user"] == GUEST

    @pytest.mark.asyncio
    async def test_guest_denied_in_untracked_channel(self):
        orch = _make_orch()
        hm = await _route(orch, channel=UNTRACKED)
        hm.assert_not_called()
        assert "not one the owner tracks" in orch.slack.post_ephemeral.call_args[0][2]

    @pytest.mark.asyncio
    async def test_guest_denied_in_review_activation(self):
        orch = _make_orch(_cfg(ACTIVATION_REVIEW))
        hm = await _route(orch)
        hm.assert_not_called()
        assert "does not answer guests" in orch.slack.post_ephemeral.call_args[0][2]

    @pytest.mark.asyncio
    async def test_guest_denied_in_off_activation(self):
        orch = _make_orch(_cfg(ACTIVATION_OFF))
        hm = await _route(orch)
        hm.assert_not_called()

    @pytest.mark.asyncio
    async def test_guest_denied_in_a_dm(self):
        orch = _make_orch(_cfg(ACTIVATION_ALWAYS))
        hm = await _route(orch, channel="D0GUESTDM")
        hm.assert_not_called()
        assert "not in DMs" in orch.slack.post_ephemeral.call_args[0][2]

    @pytest.mark.asyncio
    async def test_guest_denied_in_a_thread_that_is_not_their_own(self):
        """An owner's thread holds an owner session, and one thread has one owner."""
        orch = _make_orch()
        orch.sessions.get_session_for_thread = MagicMock(return_value="slack:9.0")
        hm = await _route(orch, is_mention=False, ts="11.0", thread="9.0")
        hm.assert_not_called()

    @pytest.mark.asyncio
    async def test_guest_followed_up_in_their_own_thread(self):
        orch = _make_orch()
        orch.sessions.get_session_for_thread = MagicMock(
            return_value=guest_session_key(GUEST, "9.0")
        )
        hm = await _route(orch, is_mention=False, ts="11.0", thread="9.0")
        hm.assert_called_once()

    @pytest.mark.asyncio
    async def test_stranger_still_refused_with_allowlist_advice(self):
        orch = _make_orch()
        hm = await _route(orch, user=STRANGER)
        hm.assert_not_called()
        assert "add you to the allowlist" in orch.slack.post_ephemeral.call_args[0][2]


# ─────────────────────── the memory-store refusal (no leak) ────────────────────


class TestGuestMemberRefusal:
    def test_unconfigured_guest_member_is_refused(self):
        cfg = _cfg(guest_agent="")
        assert resolve_guest_agent(cfg, TRACKED) == ""
        assert guest_member_configured(cfg, "") is False

    def test_a_name_absent_from_agents_is_refused(self):
        """The exact shape that would bind a guest to the owner's DEFAULT store."""
        cfg = _cfg(guest_agent="not-a-member")
        cfg.agents.pop("not-a-member", None)
        assert resolve_guest_agent(cfg, TRACKED) == "not-a-member"
        assert guest_member_configured(cfg, "not-a-member") is False
        assert guest_member_configured(cfg, DEFAULT_MEMORY_STORE) is False

    @pytest.mark.asyncio
    async def test_guest_turn_does_not_run_without_a_member(self):
        orch = _make_orch(_cfg(guest_agent=""))
        hm = await _route(orch)
        hm.assert_not_called()
        assert "guest agent for this channel is not" in orch.slack.post_ephemeral.call_args[0][2]

    def test_per_channel_guest_agent_overrides_the_global(self):
        cfg = _cfg()
        cfg.slack_channels[TRACKED].guest_agent = "channel-member"
        assert resolve_guest_agent(cfg, TRACKED) == "channel-member"
        # A channel with no override falls back to the global.
        assert resolve_guest_agent(cfg, UNTRACKED) == GUEST_MEMBER

    def test_both_paths_converge_on_the_same_refusal(self):
        """An unresolvable name refuses whether it came from the channel or global."""
        cfg = _cfg(guest_agent="")
        cfg.slack_channels[TRACKED].guest_agent = "ghost-member"
        assert resolve_guest_agent(cfg, TRACKED) == "ghost-member"
        assert guest_member_configured(cfg, "ghost-member") is False
        assert resolve_guest_agent(cfg, UNTRACKED) == ""
        assert guest_member_configured(cfg, "") is False


# ─────────────────────────── guest session isolation ──────────────────────────


class TestGuestSessionKey:
    def test_guest_key_is_not_the_owner_thread_key(self):
        from kiro_crew.messaging.link import canonical_key

        assert guest_session_key(GUEST, "5.0") != canonical_key("5.0")

    def test_two_guests_do_not_share_a_key(self):
        assert guest_session_key(GUEST, "5.0") != guest_session_key(STRANGER, "5.0")

    def test_guest_key_is_not_mistaken_for_a_bare_timestamp(self):
        from kiro_crew.messaging.link import legacy_key

        assert legacy_key(guest_session_key(GUEST, "5.0")) is None


# ─────────────────── YOLO: two independent block points ───────────────────────


class TestYoloBlockPointOne:
    """``_resolve_approval_mode`` alone, with the hooks layer out of the picture."""

    def test_owner_turn_under_yolo_is_auto(self):
        orch = _make_orch()
        with patch("kiro_crew.slack.events.is_yolo_mode", return_value=True):
            assert _resolve_approval_mode(orch) == APPROVAL_AUTO

    def test_guest_turn_under_yolo_is_interactive(self):
        orch = _make_orch()
        with patch("kiro_crew.slack.events.is_yolo_mode", return_value=True):
            assert _resolve_approval_mode(orch, guest=True) == APPROVAL_INTERACTIVE

    def test_guest_turn_is_interactive_even_with_configured_auto(self):
        orch = _make_orch()
        orch._approval_mode = APPROVAL_AUTO
        with patch("kiro_crew.slack.events.is_yolo_mode", return_value=False):
            assert _resolve_approval_mode(orch, guest=True) == APPROVAL_INTERACTIVE


class TestYoloBlockPointTwo:
    """The hooks layer alone: the owner's auto-approve list cannot grant."""

    def test_guest_hooks_drop_auto_approve_but_keep_deny(self):
        from kiro_crew.hooks import HookManager, HooksConfig

        owner_hooks = HookManager(HooksConfig(auto_approve_tools=["*"], auto_deny_tools=["curl*"]))
        guest_hooks = build_guest_hooks(owner_hooks)
        assert owner_hooks._config.auto_approve_tools == ["*"]
        assert guest_hooks._config.auto_approve_tools == []
        assert guest_hooks._config.auto_deny_tools == ["curl*"]

    def test_owner_wildcard_auto_approves_and_guest_does_not(self):
        """Same tool, same owner config, opposite verdicts from the hook gate."""
        from kiro_crew.hooks import TOOL_AUTO_APPROVE, HookManager, HooksConfig

        owner_hooks = HookManager(HooksConfig(auto_approve_tools=["*"]))
        guest_hooks = build_guest_hooks(owner_hooks)
        assert owner_hooks.on_tool_call("shell").action == TOOL_AUTO_APPROVE
        assert guest_hooks.on_tool_call("shell").action != TOOL_AUTO_APPROVE


# ─────────────────────── the guest tool allowlist ─────────────────────────────


class TestGuestSafeTools:
    @pytest.mark.parametrize("tool", sorted(GUEST_SAFE_TOOLS))
    def test_listed_tools_are_allowed(self, tool):
        assert is_guest_safe_tool(tool, "") is True

    @pytest.mark.parametrize(
        "tool",
        [
            "web_fetch",
            "shell",
            "use_aws",
            "Read",
            "Write",
            "Grep",
            "Glob",
            "spawn_run",
            "send_message",
            "cron_add",
            "memory_recall",
            "local_knowledge_search",
            "artifact_get",
            "learn_add",
        ],
    )
    def test_owner_reach_tools_are_refused(self, tool):
        assert is_guest_safe_tool(tool, "") is False

    def test_the_set_holds_exactly_one_entry(self):
        """A guest chooses words, never a destination."""
        assert GUEST_SAFE_TOOLS == frozenset({"web_search"})

    @pytest.mark.parametrize(
        "name",
        [
            "web",
            "web_",
            "web_sea",
            "web_search_admin",
            "web_search_raw",
            "unsafe_web_search",
            "my_web_search",
            "WEB_SEARCH",
            "Web_Search",
        ],
    )
    def test_only_the_exact_name_is_admitted(self, name):
        """A neighbour of the allowed name is a DIFFERENT tool and is refused.

        The match is exact, so neither a prefix of the allowed name nor a name
        containing it is admitted. A substring or case-insensitive comparison here
        would let a server expose ``web_search_raw`` and inherit the grant.
        """
        assert is_guest_safe_tool(name, "") is False
        # CONTROL: the exact name still passes, so these are refusals of the
        # neighbours rather than a matcher that admits nothing.
        assert is_guest_safe_tool("web_search", "") is True

    def test_web_fetch_is_refused_because_the_guest_picks_the_host(self):
        """``web_fetch`` takes a URL, so admitting it is admitting an arbitrary GET.

        Paired with the control below so an all-refused result cannot be mistaken
        for a matcher that matches nothing.
        """
        assert is_guest_safe_tool("web_fetch", "") is False
        # CONTROL: the gate does answer True for something, so the refusals above
        # are refusals and not a dead matcher.
        assert is_guest_safe_tool("web_search", "") is True

    @pytest.mark.parametrize(
        "name",
        [
            "web_fetch",
            "WebFetch",
            "fetch",
            "http_get",
            "curl",
        ],
    )
    def test_no_spelling_of_a_fetch_tool_is_admitted(self, name):
        """Every name a URL-taking tool could arrive under is refused."""
        assert is_guest_safe_tool(name, "") is False

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            "http://127.0.0.1:5476/api/state",
            "http://localhost:8080/admin",
            "http://[::1]:5476/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "file:///etc/passwd",
        ],
    )
    def test_a_guest_supplied_internal_url_reaches_no_tool(self, url):
        """The destination never matters, because no tool accepting one is approved.

        The gate keys on the tool identity, so a link-local or loopback address is
        unreachable for a guest whether or not the fetch builtin guards it. That is
        why the exclusion lives in the allowlist rather than in a URL filter.
        """
        for name in (f"web_fetch({url})", "web_fetch", f"web_fetch {url}"):
            assert is_guest_safe_tool(name, "") is False
        # CONTROL: a search for the same string is still permitted, so the
        # refusals above are about the tool and not about the text.
        assert is_guest_safe_tool("web_search", "") is True

    def test_read_shaped_invented_names_are_refused(self):
        for name in ("get_all_credentials", "list_env_secrets", "read_owner_memory"):
            assert is_guest_safe_tool(name, "") is False

    @pytest.mark.parametrize(
        "title",
        [
            "Running: web_search",
            "mcp__srv__web_search",
            "@srv/web_search",
            "Running: @srv/web_search",
            "Searching the web for web_search",
        ],
    )
    def test_a_wire_TITLE_spelling_is_refused(self, title):
        """The gate reads a verified identity, so a title-shaped string is not one.

        ``event.title`` is LLM-authored prose, and these are the forms a title
        arrives in. None of them is what the harness stamps as ``tool_name``, so
        each is refused -- a gate that unwrapped them would be reading the model's
        own words, and a crafted title would route onto the one safe name.
        """
        assert is_guest_safe_tool(title, "") is False
        # CONTROL: the verified spelling of the same tool passes, so this is a
        # refusal of the title FORM and not of the tool.
        assert is_guest_safe_tool("web_search", "") is True

    def test_an_mcp_served_tool_is_refused_even_under_the_allowed_name(self):
        """The guest spec mounts no MCP servers, so any server naming one is not it.

        Without this, a second server exposing its own ``web_search`` would inherit
        the grant -- the bare-name hazard ``normalize_tool_title`` documents.
        """
        assert is_guest_safe_tool("web_search", "srv") is False
        assert is_guest_safe_tool("web_search", "evil-server") is False
        # CONTROL: the same name with no server is the builtin, and passes.
        assert is_guest_safe_tool("web_search", "") is True

    def test_an_absent_identity_is_refused(self):
        """An unverified call is not a safe one, and there is no title fallback."""
        assert is_guest_safe_tool("", "") is False
        assert is_guest_safe_tool("   ", "") is False
        # Absent identity plus a server name is still absent.
        assert is_guest_safe_tool("", "srv") is False

    def test_the_gate_reads_the_verified_identity_not_the_title(self):
        """The call site must pass ``tool_name``/``mcp_server_name``, never ``title``.

        A behavioural test cannot see which FIELD the handler read, so this reads
        the call site itself. The needle is built from parts so this assertion's own
        line cannot match and report itself as the offender.
        """
        import inspect

        from kiro_crew.slack import handler as _handler

        src = inspect.getsource(_handler)
        good = "is_guest_safe_tool(" + "event.tool_name, " + "event.mcp_server_name)"
        bad = "is_guest_safe_tool(" + "event." + "title)"
        assert src.count(good) == 1, "the guest gate must key on the verified identity"
        assert bad not in src, "the guest gate must not key on the model-authored title"

    def test_guest_set_does_not_inherit_the_heartbeat_filesystem_reads(self):
        """Heartbeat is the owner's own session; a guest is a different person."""
        from kiro_crew.slack.gateway import HEARTBEAT_SAFE_TOOLS

        assert {"Read", "Grep", "Glob"} <= HEARTBEAT_SAFE_TOOLS
        assert not ({"Read", "Grep", "Glob"} & GUEST_SAFE_TOOLS)

    def test_the_guest_agent_spec_does_not_advertise_a_refused_tool(self):
        """The prompt must not offer what the gate rejects, or a turn is wasted."""
        from kiro_crew.agent import _GUEST_SYSTEM_PROMPT

        assert "web search" in _GUEST_SYSTEM_PROMPT
        assert "web fetch" not in _GUEST_SYSTEM_PROMPT


# ─────────────────── every owner control still refuses a guest ────────────────


class TestOwnerControlsRefuseGuests:
    """One case per class of control named in the contract."""

    @pytest.mark.asyncio
    async def test_bang_command_refuses_a_guest(self):
        """``!`` commands gate on the owner predicate, which a guest fails."""
        from kiro_crew.slack import handler as handler_mod

        assert handler_mod.is_allowed_user(GUEST) is False
        assert handler_mod.is_owner(GUEST) is False

    def test_interaction_buttons_gate_on_the_owner_predicate(self):
        """Every interactions.py gate reads is_allowed_user, never is_guest_user."""
        import inspect

        from kiro_crew.slack import interactions

        src = inspect.getsource(interactions)
        assert "is_guest_user" not in src
        assert src.count("is_allowed_user(user_id)") >= 8

    def test_home_tab_gates_on_the_owner_predicate(self):
        import inspect

        from kiro_crew.slack import events

        # The home-tab branch lives inside the socket-mode handler.
        src = inspect.getsource(events.init_socket_mode)
        assert "app_home_opened" in src
        assert "is_allowed_user(user)" in src
        assert "is_guest_user" not in src

    def test_presigned_dashboard_link_gates_on_the_owner_predicate(self):
        import inspect

        from kiro_crew.dashboard.handlers import messaging

        src = inspect.getsource(messaging)
        assert "is_guest_user" not in src

    def test_the_guest_predicate_reaches_only_the_message_gate(self):
        """``is_guest_user`` is consulted where messages arrive, and nowhere else."""
        import inspect

        from kiro_crew.slack import events, interactions

        assert "is_guest_user" in inspect.getsource(events._route_message)
        assert "is_guest_user" not in inspect.getsource(interactions)


# ─────────────────────── the owner keeps what they had ────────────────────────


class TestOwnerUnchanged:
    @pytest.mark.asyncio
    async def test_owner_mention_still_answered_with_no_guest_marker(self):
        orch = _make_orch()
        hm = await _route(orch, user=OWNER)
        hm.assert_called_once()
        assert hm.call_args.kwargs["guest_user"] == ""

    @pytest.mark.asyncio
    async def test_owner_in_an_untracked_channel_is_still_answered(self):
        """Tracking gates GUESTS. The owner's reach is unchanged by this feature."""
        orch = _make_orch()
        hm = await _route(orch, channel=UNTRACKED, user=OWNER)
        hm.assert_called_once()

    @pytest.mark.asyncio
    async def test_owner_dm_still_answered(self):
        orch = _make_orch(_cfg(ACTIVATION_ALWAYS))
        hm = await _route(orch, channel="D0OWNERDM", user=OWNER)
        hm.assert_called_once()

    @pytest.mark.asyncio
    async def test_owner_in_review_activation_still_dispatches(self):
        orch = _make_orch(_cfg(ACTIVATION_REVIEW))
        hm = await _route(orch, user=OWNER)
        hm.assert_called_once()

    def test_owner_predicate_is_unchanged_by_the_allowlist(self):
        """A longer allowlist does not widen who the owner predicate answers for."""
        set_allowed_users({OWNER, GUEST, STRANGER})
        assert is_allowed_user(OWNER) is True
        assert is_allowed_user(GUEST) is False
        assert is_allowed_user(STRANGER) is False


# ─────────────────────── denial text names the real cause ─────────────────────


class TestDenialText:
    def test_a_stranger_is_told_about_the_allowlist(self):
        assert "add you to the allowlist" in _denial_ephemeral(False, "")

    @pytest.mark.parametrize(
        "reason,fragment",
        [
            ("guest_dm_not_supported", "not in DMs"),
            ("guest_channel_not_tracked", "not one the owner tracks"),
            ("guest_thread_not_own", "Post a new message in the channel"),
            ("guest_member_not_configured", "guest agent for this channel is not"),
            ("guest_config_unreadable", "could not read my own configuration"),
            ("guest_backend_cannot_gate_tools", "cannot limit a guest's tools"),
            ("guest_denied_in_activation_review", "does not answer guests"),
        ],
    )
    def test_each_guest_reason_names_itself(self, reason, fragment):
        text = _denial_ephemeral(True, reason)
        assert fragment in text
        # A guest is never sent to ask for the allowlist they are already on.
        assert "add you to the allowlist" not in text

    def test_every_reason_the_code_can_set_has_its_own_text(self):
        """Enumerated from the SOURCE, not from a list a reader maintains.

        The rows above are hand-written, and that is how ``guest_config_unreadable``
        shipped with no entry: it fell through to the unauthorized text, which tells
        an already-allow-listed guest to ask the owner to add them. So this reads
        every literal the admission code assigns to ``_guest_deny_reason`` and
        requires a text for each, which makes a reason added later fail here rather
        than lie to a guest.

        The ``guest_denied_in_activation_*`` family is an f-string rather than a
        literal, so it is not enumerable this way and is covered by its own row
        above plus ``_DENIAL_TEXT_ACTIVATION``; the assertion below proves the scan
        found the literal ones.
        """
        import re

        src = Path(inspect.getfile(_route_message)).read_text(encoding="utf-8")
        assigned = set(re.findall(r'_guest_deny_reason = "([a-z_]+)"', src))
        # CONTROL: the scan resolves real assignments, so an empty offender list
        # below means the texts are present rather than that the pattern missed.
        assert "guest_dm_not_supported" in assigned
        assert len(assigned) >= 6, sorted(assigned)
        missing = sorted(r for r in assigned if r not in _GUEST_DENIAL_TEXT)
        assert not missing, (
            f"admission can set these reasons with no text of their own: {missing} -- "
            "each would fall through to the unauthorized wording and tell an "
            "allow-listed guest to ask for the allowlist they are already on"
        )


# ───────────────────────── per-store lessons ──────────────────────────────────


class TestGuestLessons:
    def test_a_named_store_has_its_own_lessons_directory(self):
        """A guest member's store is NAMED, so its lessons live outside the workspace.

        ``LessonStore`` reads ``lessons.jsonl`` under the base directory it is
        given, so distinct base directories are distinct lesson files.
        """
        from kiro_crew.memory import workspace_dir
        from kiro_crew.memory_stores import _named_store_dir

        guest_dir = _named_store_dir("guest-store")
        assert guest_dir != workspace_dir()
        assert workspace_dir() not in guest_dir.parents
        assert guest_dir.name == "guest-store"

    def test_get_lessons_for_branches_on_the_named_store(self):
        """The resolver reaches a named store's own directory, not the workspace."""
        import inspect

        from kiro_crew.context import ContextBuilder

        src = inspect.getsource(ContextBuilder.get_lessons_for)
        assert "ensure_memory_store_dir(store_name)" in src
        assert "workspace_dir_for(workspace or _DEFAULT_KEY)" in src
        # The named branch is taken whenever a store name is present.
        assert "if store_name:" in src

    def test_guest_store_is_not_the_default_store(self):
        """A config-level fact only. What the TURN resolves is ``TestGuestStoreSeam``.

        This and the assertion below say the configuration is shaped right. They do
        not say the guest turn resolves that store, and a guest turn silently
        resolving the owner's default is exactly what they missed.
        """
        cfg = _cfg()
        assert cfg.agents[GUEST_MEMBER].memory_store != DEFAULT_MEMORY_STORE

    def test_default_store_name_would_be_the_owner_store(self):
        """Why the refusal exists: the fallback store name IS the owner's."""
        assert DEFAULT_MEMORY_STORE == "default"
        cfg = _cfg(guest_agent="")
        assert guest_member_configured(cfg, DEFAULT_MEMORY_STORE) is False


# ──────────────────── guests never take the transport path ────────────────────


class TestGuestStaysNative:
    @pytest.mark.asyncio
    async def test_transport_refuses_a_guest_turn(self):
        from kiro_crew.slack.transport_dispatch import handle_message_transport

        sessions = AsyncMock()
        slack = MagicMock()
        await handle_message_transport(
            slack, sessions, TRACKED, "hi", None, "1.0", GUEST, guest_user=GUEST
        )
        # Refused before any session work.
        sessions.get_or_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_guest_dispatches_native_even_with_transport_enabled(self):
        cfg = _cfg(ACTIVATION_MENTION)
        cfg.messaging.use_transport = True
        orch = _make_orch(cfg)
        with patch(
            "kiro_crew.slack.events.handle_message_transport", new_callable=AsyncMock
        ) as transport:
            hm = await _route(orch)
        hm.assert_called_once()
        transport.assert_not_called()


# ───────── downstream: every owner-keyed mechanism the guest path reaches ──────


class TestGuestMemberMustBeIsolatedAndRestricted:
    """Membership alone proves neither half of the posture (contract item 2)."""

    def test_a_member_omitting_memory_store_is_refused(self):
        """The field DEFAULTS to ``default``, which IS the owner's own store."""
        cfg = _cfg()
        cfg.agents[GUEST_MEMBER] = MagicMock(
            memory_store=DEFAULT_MEMORY_STORE, kiro_agent=SLACK_GUEST_AGENT_NAME
        )
        assert guest_member_configured(cfg, GUEST_MEMBER) is False
        # CONTROL: the same member with a named store passes, so this refusal is
        # about the store and not about the lookup failing outright.
        cfg.agents[GUEST_MEMBER] = MagicMock(
            memory_store="guest-store", kiro_agent=SLACK_GUEST_AGENT_NAME
        )
        assert guest_member_configured(cfg, GUEST_MEMBER) is True

    def test_a_member_with_a_blank_memory_store_is_refused(self):
        cfg = _cfg()
        cfg.agents[GUEST_MEMBER] = MagicMock(memory_store="", kiro_agent=SLACK_GUEST_AGENT_NAME)
        assert guest_member_configured(cfg, GUEST_MEMBER) is False

    def test_a_member_bound_to_another_spec_is_refused(self):
        """Only the guest spec mounts no MCP servers, so only it is restricted.

        A permission-time gate cannot see a spec's ``allowedTools`` pre-approval --
        the harness raises no permission request for a pre-approved tool -- so this
        predicate is the only place a wrongly-bound spec can be stopped.
        """
        cfg = _cfg()
        for spec in ("kirocrew", "kirocrew-worker", "kirocrew-heartbeat", ""):
            cfg.agents[GUEST_MEMBER] = MagicMock(memory_store="guest-store", kiro_agent=spec)
            assert guest_member_configured(cfg, GUEST_MEMBER) is False, spec
        cfg.agents[GUEST_MEMBER] = MagicMock(
            memory_store="guest-store", kiro_agent=SLACK_GUEST_AGENT_NAME
        )
        assert guest_member_configured(cfg, GUEST_MEMBER) is True

    @pytest.mark.asyncio
    async def test_a_misconfigured_member_denies_admission_loudly(self):
        """An owner misconfiguration refuses the turn; it never downgrades it."""
        cfg = _cfg()
        cfg.agents[GUEST_MEMBER] = MagicMock(
            memory_store=DEFAULT_MEMORY_STORE, kiro_agent=SLACK_GUEST_AGENT_NAME
        )
        orch = _make_orch(cfg)
        hm = await _route(orch)
        hm.assert_not_called()
        assert "guest agent for this channel is not" in orch.slack.post_ephemeral.call_args[0][2]


class TestGuestKeywordCommands:
    """``spawn``/``run``/``cron`` run with no caller check and before any LLM turn."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        ["spawn go read the config", "run /etc/spec.yaml", "cron remove all", "sessions"],
    )
    async def test_a_guest_reaches_no_keyword_command(self, text):
        slack = MagicMock()
        slack.post_message = AsyncMock()
        subagents = MagicMock()
        runner = MagicMock()
        cron = MagicMock()
        handled = await maybe_handle_keyword_command(
            text,
            slack,
            AsyncMock(),
            TRACKED,
            "1.0",
            "1.0",
            guest_session_key(GUEST, "1.0"),
            GUEST,
            None,
            subagent_manager=subagents,
            task_runner=runner,
            cron_service=cron,
            guest_user=GUEST,
        )
        assert handled is False
        slack.post_message.assert_not_called()
        # The services are never touched, so no command ran and none was refused
        # with a reply either -- the text goes on as ordinary chat.
        assert not subagents.method_calls
        assert not runner.method_calls
        assert not cron.method_calls

    @pytest.mark.asyncio
    async def test_the_owner_still_reaches_the_sessions_command(self):
        """CONTROL: the refusal is the guest flag, not a helper that handles nothing."""
        slack = MagicMock()
        slack.post_message = AsyncMock()
        sessions = AsyncMock()
        sessions.list_sessions = MagicMock(return_value=[])
        handled = await maybe_handle_keyword_command(
            "sessions", slack, sessions, TRACKED, "1.0", "1.0", "slack:1.0", OWNER, None
        )
        assert handled is True


class TestGuestChannelHistory:
    """Guest text must not enter the buffer an OWNER turn reads as context."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("activation", [ACTIVATION_MENTION, ACTIVATION_ALWAYS])
    async def test_a_guest_message_is_not_pushed_to_channel_history(self, activation):
        orch = _make_orch(_cfg(activation))
        hm = await _route(orch)
        hm.assert_called_once()  # admitted, so this is a skip and not a denial
        orch.channel_history.push.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_owner_message_is_still_pushed(self):
        """CONTROL: the push still happens, so the guest case is a real exclusion."""
        orch = _make_orch(_cfg(ACTIVATION_ALWAYS))
        hm = await _route(orch, user=OWNER)
        hm.assert_called_once()
        orch.channel_history.push.assert_called_once()


class TestGuestThreadClaimIsInvisibleToTheOwner:
    """A guest's index claim must not re-route an OWNER's later reply."""

    def test_the_guest_key_is_recognisable(self):
        assert is_guest_session_key(guest_session_key(GUEST, "9.0")) is True
        assert is_guest_session_key("slack:9.0") is False
        assert is_guest_session_key("dashboard:chat-1") is False
        assert is_guest_session_key("") is False

    def test_an_owner_turn_does_not_adopt_a_guest_claim(self, tmp_path):
        """Driven through a REAL index, because that is where the rule lives now.

        A MagicMock ``sessions`` would return whatever it was told to and pass no
        matter what the filter does -- which is exactly how the transport path's
        ten unfiltered reads survived a green suite. The claim is written by
        ``set_slack_link`` here, not fabricated, so the test also proves the claim
        genuinely survives into the index.
        """
        sessions = _real_index(tmp_path)
        sessions.set_slack_link(guest_session_key(GUEST, "9.0"), "9.0", TRACKED)
        assert sessions.get_session_for_thread("9.0") is None

    def test_the_guest_itself_still_sees_its_own_claim(self, tmp_path):
        sessions = _real_index(tmp_path)
        sessions.set_slack_link(guest_session_key(GUEST, "9.0"), "9.0", TRACKED)
        assert sessions.get_session_for_thread("9.0", guest_user=GUEST) == guest_session_key(
            GUEST, "9.0"
        )

    def test_a_second_guest_does_not_see_the_first_guests_claim(self, tmp_path):
        """A guest sees its OWN claim only, so the filter cannot be reused sideways."""
        sessions = _real_index(tmp_path)
        sessions.set_slack_link(guest_session_key(GUEST, "9.0"), "9.0", TRACKED)
        assert sessions.get_session_for_thread("9.0", guest_user="U0GUEST99") is None

    def test_a_dashboard_claim_is_untouched_for_both(self, tmp_path):
        """CONTROL: only a GUEST claim is filtered, so real linking still works.

        Without this, a filter that hid EVERY claim would pass the three above.
        """
        sessions = _real_index(tmp_path)
        sessions.set_slack_link("dashboard:chat-7", "9.0", TRACKED)
        assert sessions.get_session_for_thread("9.0") == "dashboard:chat-7"
        assert sessions.get_session_for_thread("9.0", guest_user=GUEST) == "dashboard:chat-7"

    def test_the_handler_helper_delegates_rather_than_reimplementing(self, tmp_path):
        """``visible_thread_owner`` must carry no rule of its own.

        A filter living in ONE caller leaves every other reader of the index
        unprotected -- the Slack transport path, the dashboard mirror and the
        interaction handler each read it directly. Pinning delegation keeps the
        rule in the index, where all of them inherit it.
        """
        from kiro_crew.slack.handler import visible_thread_owner

        sessions = _real_index(tmp_path)
        sessions.set_slack_link(guest_session_key(GUEST, "9.0"), "9.0", TRACKED)
        assert visible_thread_owner(sessions, "9.0", "") is None
        assert visible_thread_owner(sessions, "9.0", GUEST) == guest_session_key(GUEST, "9.0")
        src = inspect.getsource(visible_thread_owner)
        assert "startswith" not in src and "guest-" not in src


class TestGuestMentionInSomebodyElsesThread:
    @pytest.mark.asyncio
    async def test_a_guest_mention_inside_an_owner_thread_is_refused(self):
        """The reachability premise under the whole downstream class.

        A bare ``is_mention`` admitted this, and the admitted turn is the one that
        keys to the owner's session and answers inside the owner's thread.
        """
        orch = _make_orch()
        orch.sessions.get_session_for_thread = MagicMock(return_value="slack:9.0")
        hm = await _route(orch, is_mention=True, ts="11.0", thread="9.0")
        hm.assert_not_called()
        assert "Post a new message in the channel" in orch.slack.post_ephemeral.call_args[0][2]

    @pytest.mark.asyncio
    async def test_a_guest_mention_inside_an_unclaimed_thread_is_refused(self):
        """Answering would self-link it, so the owner would find it already claimed."""
        orch = _make_orch()
        orch.sessions.get_session_for_thread = MagicMock(return_value=None)
        hm = await _route(orch, is_mention=True, ts="11.0", thread="9.0")
        hm.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_guest_mention_inside_another_guests_thread_is_refused(self):
        orch = _make_orch()
        orch.sessions.get_session_for_thread = MagicMock(
            return_value=guest_session_key("U0GUEST02", "9.0")
        )
        hm = await _route(orch, is_mention=True, ts="11.0", thread="9.0")
        hm.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_top_level_guest_mention_is_still_admitted(self):
        """CONTROL: the tightening refuses threads, not every mention."""
        orch = _make_orch()
        hm = await _route(orch, is_mention=True, ts="11.0", thread=None)
        hm.assert_called_once()


class TestGuestQueuedMessageKeepsItsIdentity:
    @pytest.mark.asyncio
    async def test_an_enqueued_guest_message_carries_guest_user(self):
        """``_dispatch_queued`` rebuilds the turn from the queue entry alone.

        Omitting the flag re-dispatched the guest's text as an OWNER turn: guest
        session key, guest hooks, guest tool gate and the YOLO refusal all key off
        this one value.
        """
        orch = _make_orch()
        orch.sessions.enqueue = MagicMock(return_value=True)
        await _route(orch)
        assert orch.sessions.enqueue.call_args.kwargs["guest_user"] == GUEST

    @pytest.mark.asyncio
    async def test_a_busy_guest_session_also_carries_it(self):
        orch = _make_orch()
        orch.sessions.enqueue = MagicMock(return_value=True)
        orch._session_tasks[guest_session_key(GUEST, "10.0")] = MagicMock()
        await _route(orch)
        assert orch.sessions.enqueue.call_args.kwargs["guest_user"] == GUEST

    @pytest.mark.asyncio
    async def test_an_owner_message_enqueues_with_no_guest_flag(self):
        """CONTROL: the value is the sender's identity, not a constant."""
        orch = _make_orch()
        orch.sessions.enqueue = MagicMock(return_value=True)
        await _route(orch, user=OWNER)
        assert orch.sessions.enqueue.call_args.kwargs["guest_user"] == ""


class TestGuestTurnThreadsItsFlagEverywhere:
    """A downstream mechanism that loses the flag loses every guest protection."""

    def test_the_compaction_replay_passes_guest_user(self):
        """The replay re-enters ``handle_message``; without the flag it is an owner.

        Needles built from parts so this assertion cannot match its own line.
        """
        import inspect

        from kiro_crew.slack import handler as _handler

        src = inspect.getsource(_handler.handle_message)
        assert src.count("guest_user=" + "guest_user") >= 1

    def test_every_handle_message_keyword_is_forwarded_by_the_replay(self):
        """Structural: the replay must forward the same keywords it received.

        A parameter added to ``handle_message`` later and not forwarded here would
        silently reset to its default on a replay -- which is exactly how
        ``guest_user`` went missing.
        """
        import inspect

        from kiro_crew.slack import handler as _handler

        params = set(inspect.signature(_handler.handle_message).parameters)
        src = inspect.getsource(_handler.handle_message)
        replay = src[src.index("_COMPACTION_RETRY_NOTICE") :]
        forwarded = {p for p in params if f"{p}=" in replay}
        # Positional/self-referential parameters the replay passes positionally or
        # constructs itself, plus the text it replays.
        exempt = {
            "slack",
            "sessions",
            "channel",
            "text",
            "thread_ts",
            "msg_ts",
            "user_id",
            "_compaction_replay",
            "context_builder",
            "cron_service",
            "conversation_log",
            "consolidator",
            "subagent_manager",
            "task_runner",
            "ctx",
        }
        missing = params - forwarded - exempt
        assert not missing, f"compaction replay drops: {sorted(missing)}"


# ───────────── end to end, through a REAL session index ───────────────────────


class TestNoTurnPathReadsTheIndexOutsideTheFilter:
    """The structural half of item 6, and the reason it is structural.

    Site-by-site filtering cannot hold: ``transport_dispatch`` reads the index ten
    times and the dashboard three more, and a green suite says nothing about any of
    them. The rule therefore lives in ``SessionMap.get_session_for_thread``, and
    what needs guarding is that nothing reaches the index AROUND that method --
    including a path added later.
    """

    @staticmethod
    def _sources():
        import kiro_crew

        root = Path(kiro_crew.__file__).parent
        return root, sorted(root.rglob("*.py"))

    def test_the_reverse_index_is_never_read_outside_its_own_module(self):
        """``_thread_to_session`` is the index. Only the index may touch it.

        A caller reading the dict directly would bypass the filter entirely, which
        no amount of care at the call sites could catch.

        Matched on the AST, not on text: the name also appears in PROSE explaining
        the index (``messaging/transport.py`` describes it in a docstring), and a
        text scan reports that as a read. An ``ast.Attribute`` node is an actual
        access, so comments and docstrings are excluded by construction rather than
        by stripping rules that have to anticipate every spelling.
        """
        import ast

        root, files = self._sources()
        offenders = []
        for path in files:
            if path.name == "session_map.py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "_thread_to_session":
                    offenders.append(f"{path.relative_to(root).as_posix()}:{node.lineno}")
        assert offenders == [], f"these read the raw thread index: {offenders}"

    def test_that_raw_index_scan_can_actually_see_a_read(self):
        """CONTROL for the scan above: it must detect a real access.

        Otherwise a wrong node type or attribute name would report every module
        clean, and the guard would be decorative.
        """
        import ast

        tree = ast.parse("def f(self):\n    return self._thread_to_session.get('1.0')\n")
        found = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr == "_thread_to_session"
        ]
        assert len(found) == 1
        # And prose must NOT be detected, which is the reason for the AST.
        prose = ast.parse('def f():\n    """mentions _thread_to_session in a docstring."""\n')
        assert not [
            n
            for n in ast.walk(prose)
            if isinstance(n, ast.Attribute) and n.attr == "_thread_to_session"
        ]

    def test_every_index_read_goes_through_the_filtering_method(self):
        """CONTROL-BEARING: the scan must find the reads before finding none bad.

        An empty offender list is only evidence if the pattern matched something.
        So the reads are counted first, and the count is asserted to be
        substantial -- the transport path alone holds nine.
        """
        root, files = self._sources()
        reads = []
        for path in files:
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                code = line.split("#", 1)[0]
                if ".get_session_for_thread(" in code:
                    reads.append((path.relative_to(root).as_posix(), i, code.strip()))
        # CONTROL: the probe reads. Without this, a typo in the pattern would
        # report a perfectly clean codebase.
        assert len(reads) >= 15, f"the scan found only {len(reads)} reads; the pattern is wrong"
        assert any(
            r[0] == "slack/transport_dispatch.py" for r in reads
        ), "the transport path's reads must be in scope -- they are the ones that leaked"
        # Every one of them resolves to the single method that filters, because it
        # is the ONLY definition and nothing reads the dict behind it (asserted
        # above). What remains is that the method really applies the rule.
        from kiro_crew.session_map import SessionMap

        body = inspect.getsource(SessionMap.get_session_for_thread)
        assert "is_guest_session_key(" in body
        assert "guest_session_key(guest_user, thread_ts)" in body

    def test_the_default_is_the_safe_direction(self):
        """A caller that says nothing must see no guest claim.

        This is what makes a path added later inherit the protection: it calls the
        method at all, and the parameter it does not pass defaults to hiding.
        """
        from kiro_crew.session_map import SessionMap

        sig = inspect.signature(SessionMap.get_session_for_thread)
        assert sig.parameters["guest_user"].default == ""
        assert sig.parameters["guest_user"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_the_manager_delegate_forwards_the_argument(self):
        """The Slack paths hold a ``SessionManager``, not a ``SessionMap``.

        A delegate that dropped the keyword would make every guest read fall back
        to the safe default -- which hides the guest's OWN claim and breaks their
        follow-up, silently.
        """
        from kiro_crew.session import SessionManager

        body = inspect.getsource(SessionManager.get_session_for_thread)
        assert "guest_user=guest_user" in body


class TestOwnerOnTheTransportPathAgainstAGuestClaim:
    """The boundary this fix is actually about.

    ``use_transport`` defaults to True (``loader.py``: ``get("use_transport", True)``)
    and ``_use_transport`` excludes only review mode and guests -- so on a default
    install a GUEST turn goes native and an OWNER turn goes TRANSPORT. A filter
    living in ``handler.py`` alone therefore protects the owner on the one path the
    owner does not normally take, while ``transport_dispatch`` reads the index ten
    times unfiltered. Asserting that the filter function returns None passes either
    way, so this drives the transport path and asserts on the STORE the turn binds,
    which is what the leak is measured in.
    """

    @staticmethod
    async def _drive(smap, *, sender):
        """Drive one transport turn and return (session_key, store) it resolved."""
        from kiro_crew.slack import transport_dispatch as td

        seen: dict[str, object] = {}

        async def _capture(_ctx, key):
            # The store a key resolves to, in the shape production resolves it:
            # a guest member's own store for a guest key, blank (the OWNER's
            # default) for anything else.
            seen["session_key"] = key
            seen["store"] = "guest-store" if is_guest_session_key(key) else ""
            return seen["store"]

        sessions = AsyncMock()
        sessions.get_session_for_thread = smap.get_session_for_thread
        sessions.set_slack_link = smap.set_slack_link
        sessions.has_session = MagicMock(return_value=False)
        sessions.is_busy = MagicMock(return_value=False)
        with (
            patch.object(td, "session_store_for_turn", new=_capture),
            patch.object(td, "_hydrate_thread_overrides", new=AsyncMock()),
            patch.object(td, "_hydrate_conv_flags", new=MagicMock()),
        ):
            with contextlib.suppress(Exception):
                await td.handle_message_transport(
                    MagicMock(),
                    sessions,
                    TRACKED,
                    "hi",
                    "100.0",
                    "101.0",
                    sender,
                )
        return seen

    @pytest.mark.asyncio
    async def test_the_owner_turn_binds_the_owners_store_not_the_guests(self, tmp_path):
        smap = _real_index(tmp_path)
        guest_key = guest_session_key(GUEST, "100.0")
        smap.set_slack_link(guest_key, "100.0", TRACKED)
        # The claim is genuinely in the index -- otherwise the assertion below
        # would pass because there was nothing to adopt.
        assert smap._thread_to_session["100.0"] == guest_key

        seen = await self._drive(smap, sender=OWNER)

        assert seen.get("session_key") is not None, "the turn never reached store resolution"
        assert seen["session_key"] != guest_key
        assert seen["session_key"] == canonical_key("100.0")
        assert seen["store"] == ""

    @pytest.mark.asyncio
    async def test_a_real_dashboard_claim_is_still_adopted(self, tmp_path):
        """CONTROL: the transport path keeps re-routing to real owners.

        Without this, a change that made ``_resolve_thread_owner`` ignore the index
        entirely would pass the test above.
        """
        smap = _real_index(tmp_path)
        smap.set_slack_link("dashboard:chat-7", "100.0", TRACKED)

        seen = await self._drive(smap, sender=OWNER)

        assert seen.get("session_key") == "dashboard:chat-7"


class TestAdmissionAndExecutionShareOneConfig:
    """Item 7's boundary: the operator moves the member BETWEEN the two reads.

    ``orch._cfg`` is a boot snapshot for every section outside
    ``_SLACK_OWNED_FIELDS``, and ``agents`` is not in that tuple. Admitting on the
    snapshot while executing against a freshly loaded config let an operator
    repoint ``kiro_agent`` or reset ``memory_store`` in between: admission passed
    on the old object and the turn ran on the new one.
    """

    @pytest.mark.asyncio
    async def test_a_member_repointed_away_from_the_guest_spec_is_refused(self):
        moved = _cfg()
        moved.agents[GUEST_MEMBER] = MagicMock(
            memory_store="guest-store", kiro_agent="some-other-agent", member_id=""
        )
        orch = _make_orch(_cfg())
        hm = await _route(orch, loaded_cfg=moved)
        hm.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_member_reset_to_the_default_store_is_refused(self):
        moved = _cfg()
        moved.agents[GUEST_MEMBER] = MagicMock(
            memory_store=DEFAULT_MEMORY_STORE, kiro_agent=SLACK_GUEST_AGENT_NAME, member_id=""
        )
        orch = _make_orch(_cfg())
        hm = await _route(orch, loaded_cfg=moved)
        hm.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unreadable_config_refuses_rather_than_falling_through(self):
        orch = _make_orch(_cfg())
        hm = await _route(orch, loaded_cfg=None)
        hm.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_admitted_turn_carries_the_object_it_was_admitted_on(self):
        """CONTROL, and the positive half: agreement still admits, and hands it down.

        Without the second assertion, execution could load its own config again and
        the window would be open with every test above still green.
        """
        loaded = _cfg()
        orch = _make_orch(_cfg())
        hm = await _route(orch, loaded_cfg=loaded)
        hm.assert_called_once()
        assert hm.call_args.kwargs["guest_cfg"] is loaded

    @pytest.mark.asyncio
    async def test_execution_uses_the_passed_object_and_does_not_reload(self):
        """The discriminating case: the passed config is GOOD, a reload is BAD.

        The tests above cannot see the difference -- ``_route`` makes
        ``KiroCrewConfig.load()`` return the same object it hands down, so reading
        either one gives the same verdict. A mutation replacing
        ``guest_cfg if guest_cfg is not None else load()`` with a bare ``load()``
        therefore survived them all.

        Here the two DISAGREE in the opposite direction from the refusal tests: the
        object admission passed is usable and the one on disk is not. Only a turn
        that honours the passed object gets past the resolution, so the refusal
        notice is the whole signal.
        """
        from kiro_crew.config.sections import KiroCrewAgentConfig
        from kiro_crew.slack import handler as _h

        good = _real_guest_cfg()
        broken = _real_guest_cfg()
        broken.agents[GUEST_MEMBER] = KiroCrewAgentConfig(
            kiro_agent="not-the-guest-spec", memory_store=DEFAULT_MEMORY_STORE
        )

        slack = AsyncMock()
        slack.post_message = AsyncMock(return_value="1.0")
        slack.start_stream = AsyncMock(return_value=None)
        slack.post_blocks = AsyncMock(return_value=None)
        sessions = AsyncMock()
        sessions.get_session_for_thread = MagicMock(return_value=None)
        sessions.has_session = MagicMock(return_value=False)
        sessions.is_cancelled = MagicMock(return_value=False)

        with patch.object(KiroCrewConfig, "load", staticmethod(lambda *a, **k: broken)):
            with contextlib.suppress(Exception):
                await _h.handle_message(
                    slack,
                    sessions,
                    TRACKED,
                    "hello",
                    None,
                    "60.0",
                    GUEST,
                    channel_agent=GUEST_MEMBER,
                    guest_user=GUEST,
                    guest_cfg=good,
                )
        refused = any(
            _h._GUEST_UNRESOLVED_NOTICE in str(c) for c in slack.post_message.call_args_list
        )
        assert refused is False, (
            "the turn refused a member the PASSED config admits, so execution "
            "reloaded its own config instead of using the object it was admitted on"
        )

    @pytest.mark.asyncio
    async def test_no_passed_object_falls_back_to_a_load_and_still_checks_it(self):
        """CONTROL: a turn arriving WITHOUT one is not trusted on an earlier verdict.

        A queued message is rebuilt from its queue entry alone, so it carries no
        config. It must load one AND re-apply the predicate to it -- otherwise the
        fix above would simply move the hole to the queue path.
        """
        from kiro_crew.config.sections import KiroCrewAgentConfig
        from kiro_crew.slack import handler as _h

        broken = _real_guest_cfg()
        broken.agents[GUEST_MEMBER] = KiroCrewAgentConfig(
            kiro_agent="not-the-guest-spec", memory_store=DEFAULT_MEMORY_STORE
        )

        slack = AsyncMock()
        slack.post_message = AsyncMock(return_value="1.0")
        slack.start_stream = AsyncMock(return_value=None)
        slack.post_blocks = AsyncMock(return_value=None)
        sessions = AsyncMock()
        sessions.get_session_for_thread = MagicMock(return_value=None)
        sessions.has_session = MagicMock(return_value=False)
        sessions.is_cancelled = MagicMock(return_value=False)

        with patch.object(KiroCrewConfig, "load", staticmethod(lambda *a, **k: broken)):
            with contextlib.suppress(Exception):
                await _h.handle_message(
                    slack,
                    sessions,
                    TRACKED,
                    "hello",
                    None,
                    "61.0",
                    GUEST,
                    channel_agent=GUEST_MEMBER,
                    guest_user=GUEST,
                    guest_cfg=None,
                )
        refused = any(
            _h._GUEST_UNRESOLVED_NOTICE in str(c) for c in slack.post_message.call_args_list
        )
        assert refused is True, "a turn with no passed config must check the one it loads"


class TestGuestGateConsultsTheDenyRungs:
    """The guest gate is terminal, so a deny it does not consult never runs.

    ``build_guest_hooks`` keeps the owner's ``auto_deny_tools`` and sensitive-path
    deny, on the stated ground that "a deny can only narrow what a guest reaches".
    While the approval came first, that retained deny was never consulted on a
    guest turn at all.
    """

    def test_the_gate_consults_the_hook_before_it_approves(self):
        from kiro_crew.slack import handler as _h

        src = inspect.getsource(_h.handle_message)
        gate = src.index("if guest_user:\n", src.index("EVENT_PERMISSION_REQUEST"))
        approve = src.index("inc_tool_auto_approved", gate)
        consult = src.index("_turn_hooks.on_tool_call(", gate)
        assert consult < approve, "the hook is consulted after the approval, so never for a guest"

    def test_a_hook_deny_is_honoured_and_an_auto_approve_is_not(self):
        """The two layers compose ONE way: either alone refuses, neither widens."""
        from kiro_crew.slack import handler as _h

        src = inspect.getsource(_h.handle_message)
        gate = src.index("if guest_user:\n", src.index("EVENT_PERMISSION_REQUEST"))
        block = src[gate : src.index("inc_tool_auto_approved", gate)]
        assert "_guest_hook_denied = _guest_hook_result.action == TOOL_DENY" in block
        assert "and not _guest_hook_denied" in block
        # CONTROL: the guest branch must not honour an auto-approve, which would
        # let the owner's globs widen the guest list.
        assert "TOOL_AUTO_APPROVE" not in block

    def test_a_hook_denial_is_audited_as_such(self):
        from kiro_crew.slack import handler as _h

        src = inspect.getsource(_h.handle_message)
        assert "denied_by_hooks" in src


class TestAllowlistGrantPersistsBeforeItPublishes:
    def test_the_durable_write_precedes_the_live_set(self):
        from kiro_crew.slack import interactions

        src = inspect.getsource(interactions._handle_allowlist)
        approve = src.index("ACTION_ALLOWLIST_APPROVE")
        persist = src.index("run_config_write(persist_allowed_user", approve)
        publish = src.index("_allowed_users.add(", approve)
        assert persist < publish, "a failed durable write would leave an unrecorded grant"

    def test_the_audit_row_follows_both(self):
        """CONTROL: the SEL row is still reached, so ordering did not drop it."""
        from kiro_crew.slack import interactions

        src = inspect.getsource(interactions._handle_allowlist)
        approve = src.index("ACTION_ALLOWLIST_APPROVE")
        publish = src.index("_allowed_users.add(", approve)
        assert src.index("slack.allowlist.approve", approve) > publish


class TestReNominatingAGuestOffersKeepNotDeny:
    def test_the_prompt_predicate_is_the_membership_one(self):
        from kiro_crew.slack import allowlist

        src = inspect.getsource(allowlist.prompt_allowlist)
        assert "already = is_guest_user(user_id)" in src
        # CONTROL: the owner-standing predicate must not decide this prompt --
        # it is owner-only, so an existing guest read as "not yet allowed" and the
        # prompt offered Deny, which un-persists the grant.
        assert "already = is_allowed_user(user_id)" not in src

    @pytest.mark.asyncio
    async def test_an_existing_guest_gets_keep_and_remove(self):
        from kiro_crew.slack import allowlist

        slack = MagicMock()
        slack.open_dm = AsyncMock(return_value="D1")
        slack.post_message = AsyncMock()
        with patch.object(allowlist, "_send_prompt", new_callable=AsyncMock) as prompt:
            await allowlist.prompt_allowlist(slack, OWNER, GUEST)
        labels = [a for a in prompt.call_args[0] if isinstance(a, str)]
        assert any("Keep" in a for a in labels)
        assert not any("Deny" in a for a in labels)

    @pytest.mark.asyncio
    async def test_an_unknown_user_still_gets_allow_and_deny(self):
        """CONTROL: nomination of a NEW user is unchanged."""
        from kiro_crew.slack import allowlist

        with patch.object(allowlist, "_send_prompt", new_callable=AsyncMock) as prompt:
            await allowlist.prompt_allowlist(MagicMock(), OWNER, "U0NEWCOMER")
        labels = [a for a in prompt.call_args[0] if isinstance(a, str)]
        assert any("Allow" in a for a in labels)
        assert not any("Keep" in a for a in labels)


class TestIdentityGlobalsAreRestoredByAFixture:
    def test_the_gateway_test_file_has_a_file_scoped_teardown(self):
        """The leak crossed tests on the xdist worker, so the scope is the FILE.

        A ``finally`` at the two lines that set the globals would fix only the test
        somebody had already noticed; the next setter reopens it.
        """
        repo = Path(__file__).resolve().parents[1]
        src = (repo / "test" / "test_slack_gateway.py").read_text(encoding="utf-8")
        fixture = src.index("def _restore_slack_identity_globals")
        assert "@pytest.fixture(autouse=True)" in src[max(0, fixture - 200) : fixture]
        body = src[fixture : fixture + 1400]
        assert "finally:" in body
        assert "set_owner_id(_saved_owner)" in body
        assert "set_allowed_users(_saved_allowed)" in body
        # The fixture must be at MODULE level, not nested in a class, or it covers
        # only that class's tests and the rest of the file stays exposed.
        assert "\n    def _restore_slack_identity_globals" not in src
        assert "\ndef _restore_slack_identity_globals" in src


class TestTheGuestStoreRefusalReadsTheRightAttribute:
    """The defect a real run found, and the reason no test could.

    ``handle_message``'s guest resolution wraps ``resolve_member_execution`` with
    the admission re-check and the default-store refusal. An exception raised
    inside that wrapper is caught by the SAME ``except`` that implements the
    intended refusal -- so a wrapper that raised on every config refused every
    guest turn while every refusal test still passed, because a refusal test can
    only assert that the turn did not run.

    It did raise on every config: ``ExecutionContext.store`` is a
    ``MemoryStoreRef``, not a string, so ``(execution.store or "").strip()`` is an
    ``AttributeError``. Asserted here on the REAL object, because the type is the
    whole defect, plus the attribute the wrapper must read.

    A real driven turn is what surfaced it; that run is the end-to-end evidence.
    This is the regression pin.
    """

    def test_the_store_is_a_ref_whose_name_lives_on_store_id(self):
        from kiro_crew.execution_context import MemoryStoreRef, resolve_member_execution

        execution = resolve_member_execution(_real_guest_cfg(), GUEST_MEMBER)
        assert isinstance(execution.store, MemoryStoreRef)
        # The trap: it is NOT a string, so every string operation on it raises.
        assert not isinstance(execution.store, str)
        assert not hasattr(execution.store, "strip")
        assert execution.store.store_id == "guest-store"

    def test_the_guest_refusal_reads_store_id(self):
        from kiro_crew.slack import handler as _h

        src = inspect.getsource(_h.handle_message)
        assert 'getattr(execution.store, "store_id", "")' in src
        # CONTROL: the broken spelling must not come back. Written in parts so
        # this assertion's own source cannot match itself.
        broken = "(execution." + "store or " + '"").strip()'
        assert broken not in src

    def test_a_defaulted_member_still_refuses_on_a_real_config(self):
        """The refusal is reachable for the RIGHT reason, on real objects.

        ``MemoryStoreRef`` rejects a member on the Global store outright, so the
        resolution raises before the wrapper's own check -- either way the turn is
        refused, and the refusal is a real property of the configuration rather
        than an artifact of reading the wrong attribute.
        """
        from kiro_crew.config.sections import KiroCrewAgentConfig
        from kiro_crew.execution_context import resolve_member_execution

        cfg = _real_guest_cfg()
        cfg.agents[GUEST_MEMBER] = KiroCrewAgentConfig(
            kiro_agent=SLACK_GUEST_AGENT_NAME, memory_store=DEFAULT_MEMORY_STORE
        )
        try:
            execution = resolve_member_execution(cfg, GUEST_MEMBER)
        except Exception:
            return  # refused at resolution, which is the stronger outcome
        assert (getattr(execution.store, "store_id", "") or "") == DEFAULT_MEMORY_STORE


class TestGuestEndToEndThroughARealSessionIndex:
    """The component tests above gate each site in isolation, with a mocked index.

    That is precisely why a whole class of defects survived them: the sites agree
    with their own mocks and disagree with each other. These drive the REAL
    ``SessionMap``, so the self-derived/``setdefault`` behaviour that decides who
    holds a contested thread is the production one.
    """

    @staticmethod
    def _orch_with_real_index():
        from kiro_crew.session_map import SessionMap

        smap = SessionMap()
        orch = _make_orch()
        orch.sessions.get_session_for_thread = smap.get_session_for_thread
        orch.sessions.set_slack_link = smap.set_slack_link
        orch.sessions.has_session = MagicMock(return_value=False)
        return orch, smap

    @pytest.mark.asyncio
    async def test_mention_then_self_link_then_follow_up(self):
        orch, smap = self._orch_with_real_index()

        # 1. A top-level guest mention is admitted and keyed to the guest.
        hm = await _route(orch, is_mention=True, ts="100.0", thread=None)
        hm.assert_called_once()
        assert hm.call_args.kwargs["guest_user"] == GUEST
        expected = guest_session_key(GUEST, "100.0")

        # 2. The self-link the turn performs, against the real index. Asserted on
        # the RAW index as well as through the filtered read: the filtered read
        # alone could not tell "the claim landed" from "the filter echoed it", and
        # step 3 below only means something if the claim is genuinely stored.
        smap.set_slack_link(expected, "100.0", TRACKED)
        assert smap._thread_to_session["100.0"] == expected
        assert smap.get_session_for_thread("100.0", guest_user=GUEST) == expected
        assert smap.get_session_for_thread("100.0") is None

        # 3. A bare follow-up in that thread is admitted, decided by the real index.
        hm2 = await _route(orch, is_mention=False, ts="101.0", thread="100.0")
        hm2.assert_called_once()
        assert hm2.call_args.kwargs["guest_user"] == GUEST

    @pytest.mark.asyncio
    async def test_a_second_guest_cannot_join_the_first_guests_thread(self):
        orch, smap = self._orch_with_real_index()
        smap.set_slack_link(guest_session_key(GUEST, "100.0"), "100.0", TRACKED)
        set_allowed_users({OWNER, GUEST, "U0GUEST02"})
        hm = await _route(orch, user="U0GUEST02", is_mention=True, ts="101.0", thread="100.0")
        hm.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_guest_mention_in_a_thread_the_owner_used_is_refused(self):
        orch, smap = self._orch_with_real_index()
        # The owner's own self-link, exactly as an owner turn writes it.
        smap.set_slack_link(canonical_key("200.0"), "200.0", TRACKED)
        assert smap.get_session_for_thread("200.0") == canonical_key("200.0")
        hm = await _route(orch, is_mention=True, ts="201.0", thread="200.0")
        hm.assert_not_called()
        assert "Post a new message in the channel" in orch.slack.post_ephemeral.call_args[0][2]

    @pytest.mark.asyncio
    async def test_the_owner_keeps_their_own_key_in_a_guest_claimed_thread(self):
        """Both key shapes are self-derived, so the guest's claim is NOT evicted.

        That is what makes the filter load-bearing rather than cosmetic: the index
        genuinely still answers with the guest's key here.
        """
        from kiro_crew.slack.handler import visible_thread_owner

        orch, smap = self._orch_with_real_index()
        guest_key = guest_session_key(GUEST, "100.0")
        smap.set_slack_link(guest_key, "100.0", TRACKED)

        # The owner's later self-link does NOT displace it (setdefault branch).
        # Read on the RAW index: the whole point is that the claim is still THERE,
        # which a filtered read cannot distinguish from it having been evicted.
        smap.set_slack_link(canonical_key("100.0"), "100.0", TRACKED)
        assert smap._thread_to_session["100.0"] == guest_key

        # So the owner turn must decline to adopt it, and the guest still sees it.
        assert visible_thread_owner(smap, "100.0", "") is None
        assert visible_thread_owner(smap, "100.0", GUEST) == guest_key

    @pytest.mark.asyncio
    async def test_an_owner_mention_in_a_guest_thread_is_still_answered(self):
        """CONTROL: the owner loses no capability, they only keep their own key."""
        orch, smap = self._orch_with_real_index()
        smap.set_slack_link(guest_session_key(GUEST, "100.0"), "100.0", TRACKED)
        hm = await _route(orch, user=OWNER, is_mention=True, ts="101.0", thread="100.0")
        hm.assert_called_once()
        assert hm.call_args.kwargs["guest_user"] == ""


class TestRouteLoopGuardsAreAtTheirSites:
    """Structural, and deliberately so.

    These three guards sit inside ``handle_message``'s route loop, past
    ``get_or_create`` and the provider handshake, so reaching them behaviourally
    needs a live ACP stack. The repo already gates one security property this way
    (``test_hooks.py`` scans for the shared kwargs extraction), so these follow that
    shape: assert the guard is spelled at its site. Each needle is built from parts,
    so these assertions' own lines cannot match and report themselves.
    """

    @staticmethod
    def _src():
        import inspect

        from kiro_crew.slack import handler as _handler

        return inspect.getsource(_handler.handle_message)

    def test_the_reroute_block_excludes_a_guest_turn(self):
        """It hands the turn to the thread's owner, which for a guest is not them.

        Without the guard, ``candidate_key`` becomes the canonical or owner key and
        ``session_store_for_turn`` then resolves the OWNER's memory store.
        """
        src = self._src()
        guard = "if not route_pinned and " + "not guest_user:"
        assert src.count(guard) == 2, "both halves of the reroute must exclude a guest"
        assert ("if not route_pinned:" + "\n") not in src, "an unguarded reroute half remains"

    def test_the_linked_thread_intercept_excludes_a_guest_turn(self):
        """It delivers into the owner's dashboard slot and returns before the gate."""
        src = self._src()
        assert ("if not guest_user and await " + "maybe_route_linked_thread(") in src

    def test_the_second_agent_resolution_excludes_a_guest_turn(self):
        """It reads the thread override map and then the OWNER's default agent."""
        src = self._src()
        # The guarded form, and no unguarded re-resolution left behind.
        assert ("if not guest_user:" + "\n            _agent = (") in src
        bad = "_agent = _thread_agents.get(session_key) or channel_agent"
        assert src.count(bad) == 1, "only the pre-gate resolution may be unguarded"

    def test_no_raw_index_read_survives_inside_handle_message(self):
        """Every read goes through the filter, which is what the docstring claims.

        The three that did NOT were found by review, not by a test: an owner's
        OPTIONS control was filed under a guest's key and the guest's next message
        expired it. This is the standing guard for that whole class, so a fourth
        raw read cannot be added silently. Needles built from parts so this
        assertion's own line cannot match.
        """
        import inspect

        from kiro_crew.slack import handler as _handler

        src = inspect.getsource(_handler.handle_message)
        raw = "sessions.get_session_for_thread(" + "reply_ts)"
        filtered = "visible_thread_owner(sessions, " + "reply_ts, guest_user)"
        assert raw not in src, "a raw thread-index read bypasses the guest filter"
        # CONTROL: the filtered form is present several times, so the absence above
        # is a routed read rather than a function that stopped reading the index.
        assert src.count(filtered) >= 4

    def test_the_guest_turn_mirrors_to_no_dashboard_slot(self):
        """The clearing assignment must be its OWN statement, right after the guard.

        ``linked_session_key = None if route_pinned else thread_owner_key`` CONTAINS
        the bare assignment as a substring, so a plain containment check matches that
        line and passes whether or not a guest is cleared at all. Anchor on the
        conditional line, then require the bare statement after it.
        """
        src = self._src()
        anchor = "linked_session_key = None if route_pinned else thread_owner_key"
        tail = src[src.index(anchor) + len(anchor) :]
        assert tail.lstrip().startswith("if guest_user:"), "the guard must follow the assignment"
        assert "\n        linked_session_key = None\n" in tail, "a guest turn must clear it"


# ─────────────── seam tests: assert on what CROSSES the seam ───────────────────
#
# Every test above this line asserts a value inside one layer. Three findings got
# through 130 of them because the layer BELOW re-derives the value asserted:
# `channel_agent` is a member key the provider never sees, a config `memory_store`
# is not what a session key resolves to, and `GUEST_SAFE_TOOLS` is not what the
# agent is allowed to see. These assert the payload at the boundary instead.


def _real_guest_cfg():
    """A real config whose guest member genuinely resolves (no mocks inside)."""
    from kiro_crew.config.sections import KiroCrewAgentConfig, MemoryStoreConfig

    cfg = KiroCrewConfig(
        slack_channels={TRACKED: ChannelConfig(activation=ACTIVATION_MENTION)},
        messaging=MessagingConfig(use_transport=False),
    )
    cfg.slack.guest_agent = GUEST_MEMBER
    cfg.memory_stores["guest-store"] = MemoryStoreConfig()
    cfg.agents[GUEST_MEMBER] = KiroCrewAgentConfig(
        kiro_agent=SLACK_GUEST_AGENT_NAME, memory_store="guest-store"
    )
    return cfg


class TestGuestAgentSeam:
    """What agent name reaches the session layer, and does it survive below it.

    The strongest form would capture `SessionDeps.session_factory`'s `agent` kwarg,
    but nothing in the suite injects that factory and building the injection is more
    new scaffolding than this round should carry. So the path is closed in three
    assertions with no un-asserted gap between them: the member resolves to the guest
    TEMPLATE, the re-deriving layer leaves that template alone, and the template
    names a file that exists. Stated plainly because two of the three findings this
    replaces were exactly a gap between two individually-true assertions.
    """

    def test_the_member_resolves_to_the_guest_template_not_its_own_name(self):
        from kiro_crew.execution_context import resolve_member_execution

        execution = resolve_member_execution(_real_guest_cfg(), GUEST_MEMBER)
        assert execution.template_id == SLACK_GUEST_AGENT_NAME
        # The member KEY is a different string from the template, and only the
        # template names a spec the backend can resolve.
        assert execution.template_id != GUEST_MEMBER

    def test_the_re_deriving_layer_does_not_substitute_the_guest_template(self):
        """`session_allocation` swaps `agent` for `preparation.template` when a
        revision exists, so a template that survives that call is the one spawned.
        """
        from kiro_crew.session_capabilities import prepare_runtime

        preparation = prepare_runtime(SLACK_GUEST_AGENT_NAME, None, None)
        # No revision means the caller's agent is passed through untouched.
        assert not preparation.revision
        if preparation.template:
            assert preparation.template == SLACK_GUEST_AGENT_NAME

    def test_the_template_name_resolves_to_a_file_that_exists_after_install(self, tmp_path):
        """A name with no spec on disk fails every turn with "not installed"."""
        from kiro_crew import agent as agent_mod
        from kiro_crew.agent_files import SLACK_GUEST_AGENT_FILENAME

        with patch.object(agent_mod, "kiro_agents_dir_path", lambda: tmp_path):
            agent_mod._install_slack_guest_agent()
        installed = tmp_path / SLACK_GUEST_AGENT_FILENAME
        assert installed.is_file()
        assert installed.stem == SLACK_GUEST_AGENT_NAME

    def test_the_guest_branch_hands_over_a_template_and_not_a_member_key(self):
        """Structural backstop on the one line that chose between the two.

        Needles built from parts so this assertion cannot match its own line.
        """
        import inspect

        from kiro_crew.slack import handler as _handler

        src = inspect.getsource(_handler.handle_message)
        assert ("_agent = _guest_execution." + "template_id") in src
        assert ("_agent = " + "channel_agent or None") not in src


class TestGuestStoreSeam:
    """What store the GUEST SESSION KEY resolves to, read back as production reads it.

    `store_of_session` is the function the turn's store resolution goes through, so
    these call it rather than inspecting config. The first case is the finding: a
    fresh guest key with nothing bound resolves BLANK, and blank is the owner's own
    memory and the operator's global lessons.
    """

    @staticmethod
    def _execution():
        from kiro_crew.execution_context import resolve_member_execution

        return resolve_member_execution(_real_guest_cfg(), GUEST_MEMBER)

    def test_an_unbound_guest_key_resolves_a_blank_store(self):
        """The bug, pinned as the reason the bind exists. Blank IS the owner's."""
        from kiro_crew.context import store_of_session

        key = guest_session_key(GUEST, "500.0")
        assert store_of_session(None, key) == ""

    def test_binding_the_member_execution_makes_the_guest_key_resolve_its_store(self):
        from kiro_crew.context import store_of_session
        from kiro_crew.execution_context import bind_session_execution

        key = guest_session_key(GUEST, "501.0")
        bind_session_execution(key, self._execution())
        assert store_of_session(None, key) == "guest-store"

    def test_the_first_turn_is_the_case_that_was_blank(self):
        """A FRESH key, bound once, resolves the member's store immediately.

        The first turn is the one with no prior record to fall back on, so it is the
        turn the missing bind silently redirected to the owner's memory.
        """
        from kiro_crew.context import store_of_session
        from kiro_crew.execution_context import bind_session_execution

        key = guest_session_key(GUEST, "502.0")
        assert store_of_session(None, key) == ""  # before: the leak
        bind_session_execution(key, self._execution())
        assert store_of_session(None, key) == "guest-store"  # after: the member's

    def test_the_bound_store_is_not_the_owner_default(self):
        from kiro_crew.context import store_of_session
        from kiro_crew.execution_context import bind_session_execution

        key = guest_session_key(GUEST, "503.0")
        bind_session_execution(key, self._execution())
        resolved = store_of_session(None, key)
        assert resolved != DEFAULT_MEMORY_STORE
        assert resolved != ""

    def test_the_turn_binds_before_it_resolves(self):
        """Ordering is the whole fix: resolving first reads the blank.

        Needles built from parts so this assertion cannot match its own line.
        """
        import inspect

        from kiro_crew.slack import handler as _handler

        src = inspect.getsource(_handler.handle_message)
        bind_at = src.index("bind_session_execution" + ", candidate_key")
        resolve_at = src.index("await session_store_for_turn" + "(context_builder, candidate_key)")
        assert bind_at < resolve_at


class TestGuestToolPayloadSeam:
    """What tool payload the PROVIDER receives, which is the spec file on disk.

    The provider reads the agent spec; it never sees `GUEST_SAFE_TOOLS`. So these
    assert the written file. An empty `tools` list mounts nothing, which made the
    gate guard a tool the model could never call.
    """

    @staticmethod
    def _installed(tmp_path):
        import json

        from kiro_crew import agent as agent_mod
        from kiro_crew.agent_files import SLACK_GUEST_AGENT_FILENAME

        with patch.object(agent_mod, "kiro_agents_dir_path", lambda: tmp_path):
            agent_mod._install_slack_guest_agent()
        return json.loads((tmp_path / SLACK_GUEST_AGENT_FILENAME).read_text())

    def test_the_spec_mounts_exactly_web_search(self, tmp_path):
        spec = self._installed(tmp_path)
        assert spec["tools"] == ["web_search"]

    def test_the_spec_does_not_pre_approve_it(self, tmp_path):
        """`allowedTools` is the one path that never reaches the PreToolUse gate.

        Absent is required, not incidental: pre-approving the tool would mean
        `is_guest_safe_tool` never runs, and the gate is the terminal decision.
        """
        spec = self._installed(tmp_path)
        assert "allowedTools" not in spec
        assert "permissions" not in spec

    def test_the_spec_mounts_no_mcp_server(self, tmp_path):
        spec = self._installed(tmp_path)
        assert spec["mcpServers"] == {}

    def test_the_mounted_set_matches_what_the_gate_admits(self, tmp_path):
        """The two must agree, in both directions.

        A mounted tool the gate refuses wastes a turn; a gate entry that is not
        mounted can never be called. Either way one of them is wrong.
        """
        spec = self._installed(tmp_path)
        assert set(spec["tools"]) == set(GUEST_SAFE_TOOLS)

    def test_an_empty_tools_list_would_mount_nothing(self, tmp_path):
        """Why `[]` was the bug, pinned against the repo's own precedent.

        `kirocrew-lite` ships `tools: []` precisely to have no tools, so the empty
        list is not "unset" and does not inherit a default.
        """
        import inspect

        from kiro_crew import agent as agent_mod

        lite = inspect.getsource(agent_mod._install_lite_agent_fallback)
        assert '"tools": []' in lite
        spec = self._installed(tmp_path)
        assert spec["tools"] != []


class TestGuestOwnerTelemetry:
    """`status` answers with no LLM turn, so no tool gate exists to stop it."""

    @pytest.mark.asyncio
    async def test_a_guest_is_refused_the_status_keyword(self):
        from kiro_crew.slack import handler as _handler

        slack = MagicMock()
        slack.post_message = AsyncMock()
        with patch.object(_handler, "sel") as _sel:
            handled = await _handler.maybe_handle_keyword_command(
                "status",
                slack,
                AsyncMock(),
                TRACKED,
                "1.0",
                "1.0",
                guest_session_key(GUEST, "1.0"),
                GUEST,
                None,
                guest_user=GUEST,
            )
        assert handled is False  # the shared helper refuses every branch
        del _sel

    def test_every_text_branch_above_the_shared_helper_checks_its_caller(self):
        """The sweep, as a standing guard rather than a one-off reading.

        `status` sat above `maybe_handle_keyword_command`, so the helper's guest
        refusal never ran for it. This enumerates every branch in that region whose
        condition reads the inbound text and requires each to name a caller check,
        so the next branch added there cannot repeat it.
        """
        import inspect
        import re

        from kiro_crew.slack import handler as _handler

        lines = inspect.getsource(_handler.handle_message).splitlines()
        stop = next(i for i, l in enumerate(lines) if "maybe_handle_keyword_command(" in l)
        pat = re.compile(r"^\s*(el)?if\b.*\b(text|_cmd_text|_stripped)\b.*(==|startswith\()")

        def ind(s):
            return len(s) - len(s.lstrip())

        open_branches = []
        for i in range(stop):
            line = lines[i]
            if not pat.search(line):
                continue
            j = i + 1
            while j < stop and (
                not lines[j].strip()
                or lines[j].lstrip().startswith("#")
                or ind(lines[j]) > ind(line)
            ):
                j += 1
            body = "\n".join(lines[i:j])
            checked = (
                ("is_owner(" in body) or ("is_allowed_user(" in body) or ("guest_user" in body)
            )
            if not checked:
                open_branches.append((i + 1, line.strip()[:70]))
        assert not open_branches, f"text branches with no caller check: {open_branches}"
        # CONTROL: the sweep found branches at all, so an empty result is a pass
        # rather than a pattern that matched nothing.
        assert sum(1 for line in lines[:stop] if pat.search(line)) >= 3


class TestComposedInterceptorRefusal:
    """A composed gate can mint a presigned dashboard link, so a guest is refused."""

    def test_the_shipped_default_is_not_composed(self):
        from kiro_crew.slack.events import composed_interceptor_registered

        assert composed_interceptor_registered() is False

    def test_a_gate_overriding_intercept_message_is_composed(self):
        from kiro_crew.platform.defaults import DefaultSlackEnterpriseGate
        from kiro_crew.platform.interfaces import InterceptDecision
        from kiro_crew.slack import events as _events

        class _Composed(DefaultSlackEnterpriseGate):
            def intercept_message(self, orch, **kw):
                return InterceptDecision.REDIRECTED

        ctx = MagicMock()
        ctx.slack_gate = _Composed()
        with patch.object(_events, "current_context", lambda: ctx):
            assert _events.composed_interceptor_registered() is True

    def test_a_subclass_that_inherits_the_default_is_not_composed(self):
        from kiro_crew.platform.defaults import DefaultSlackEnterpriseGate
        from kiro_crew.slack import events as _events

        class _Inherits(DefaultSlackEnterpriseGate):
            pass

        ctx = MagicMock()
        ctx.slack_gate = _Inherits()
        with patch.object(_events, "current_context", lambda: ctx):
            assert _events.composed_interceptor_registered() is False

    def test_an_unreadable_gate_is_treated_as_composed(self):
        """Deny-by-default, matching the seam's own fail-closed contract."""
        from kiro_crew.slack import events as _events

        def _boom():
            raise RuntimeError("no context")

        with patch.object(_events, "current_context", _boom):
            assert _events.composed_interceptor_registered() is True

    @pytest.mark.asyncio
    async def test_a_guest_is_refused_while_a_composed_gate_is_registered(self):
        from kiro_crew.slack import events as _events

        orch = _make_orch()
        with patch.object(_events, "composed_interceptor_registered", lambda: True):
            hm = await _route(orch)
        hm.assert_not_called()
        assert "access gate that guests are not routed through" in (
            orch.slack.post_ephemeral.call_args[0][2]
        )

    @pytest.mark.asyncio
    async def test_the_owner_is_unaffected_by_that_refusal(self):
        """CONTROL: the refusal is scoped to guests, not to the channel."""
        from kiro_crew.slack import events as _events

        orch = _make_orch()
        with patch.object(_events, "composed_interceptor_registered", lambda: True):
            hm = await _route(orch, user=OWNER)
        hm.assert_called_once()


class TestRevokingAGuestTakesEffectWithoutARestart:
    """The config applier is the boundary, and ``is_guest_user`` is the answer.

    A grant that only a restart can withdraw is not a grant the operator
    controls. The set behind the predicate was built once in
    ``GatewayOrchestrator.__init__`` and no applier reconciled it, so an operator
    deleting a guest from ``slack.allowed_users`` changed the document and
    nothing else -- the revoked guest kept being admitted.

    So these drive ``_on_slack_config_change`` (the real applier, with the real
    ``set_allowed_users`` behind it) and ask the PREDICATE. Asserting that some
    intermediate set changed would pass just as well against the broken code
    reading a different cache, which is the mistake that let three defects
    through 178 green tests.
    """

    GUEST_TWO = "U0GUEST02"

    def _orch(self):
        """A Slack-less orchestrator owning {OWNER, GUEST}, pushed as boot does.

        ``events.py`` publishes ``orch._allowed_users`` into the handler global at
        startup, so the pre-state is established the same way rather than by
        writing the global directly.
        """
        from kiro_crew.slack.gateway import GatewayOrchestrator

        cfg = KiroCrewConfig()
        with patch.object(cfg, "load_credentials", return_value={"KIROCREW_OWNER_ID": OWNER}):
            orch = GatewayOrchestrator(cfg, no_dashboard=True, no_crons=True, no_open=True)
        orch._allowed_users.clear()
        orch._allowed_users.update({OWNER, GUEST})
        set_allowed_users(orch._allowed_users)
        return orch

    @staticmethod
    def _revoke_to(*slack_ids: str):
        """One reload whose ``slack.allowed_users`` holds exactly *slack_ids*."""
        from _hot_reload_helpers import cfg_with, change

        cfg = cfg_with("slack", allowed_users=[{"slack_id": u} for u in slack_ids])
        return change(cfg, "slack.allowed_users")

    def test_a_revoked_guest_stops_being_admitted(self):
        orch = self._orch()
        # PRE-STATE, so a False below cannot be the fixture's doing.
        assert is_guest_user(GUEST) is True
        asyncio.run(orch._on_slack_config_change(self._revoke_to()))
        assert is_guest_user(GUEST) is False

    def test_the_add_direction_lands_too(self):
        """Same applier, opposite direction -- an additive update would pass the
        revocation test only by never shrinking, so both halves are pinned."""
        orch = self._orch()
        assert is_guest_user(self.GUEST_TWO) is False
        asyncio.run(orch._on_slack_config_change(self._revoke_to(GUEST, self.GUEST_TWO)))
        assert is_guest_user(GUEST) is True
        assert is_guest_user(self.GUEST_TWO) is True

    def test_the_owner_survives_a_revocation_of_everyone(self):
        """The owner is in the set by construction, not by allow-list membership.

        Recomputing from the document alone would evict the owner, and every
        owner control keys off ``is_owner`` -- but the set still carries them, so
        a rebuild that dropped the owner would silently change admission.
        """
        orch = self._orch()
        asyncio.run(orch._on_slack_config_change(self._revoke_to()))
        assert OWNER in orch._allowed_users
        assert is_allowed_user(OWNER) is True
        assert is_guest_user(OWNER) is False

    def test_both_caches_stay_one_object(self):
        """The orchestrator set, the handler global and the modal's set are one.

        Rebinding ``self._allowed_users`` instead of mutating it would leave the
        Slack-native modal editing an orphan.
        """
        from kiro_crew.slack import handler as _handler

        orch = self._orch()
        held = orch._allowed_users
        asyncio.run(orch._on_slack_config_change(self._revoke_to(self.GUEST_TWO)))
        assert orch._allowed_users is held
        assert _handler._allowed_users is held

    def test_the_push_reconverges_a_handler_global_that_drifted(self):
        """The explicit push is load-bearing, not redundant with the in-place edit.

        Normally the handler global and ``orch._allowed_users`` are ONE object, so
        mutating in place moves admission on its own and the push looks
        decorative. They come apart whenever the global points at some other
        set -- a second orchestrator built in the same process leaves the global
        on the previous one's set, and then the new orchestrator's in-place edit
        moves nothing a gate reads.

        So this de-aliases them first. Admission must follow the applied config,
        which is only true because the applier pushes.
        """
        from kiro_crew.slack import handler as _handler

        orch = self._orch()
        # A stale set from "another orchestrator": equal contents, other object.
        stale = {OWNER, GUEST}
        set_allowed_users(stale)
        assert _handler._allowed_users is not orch._allowed_users
        assert is_guest_user(GUEST) is True

        asyncio.run(orch._on_slack_config_change(self._revoke_to()))

        assert is_guest_user(GUEST) is False
        assert _handler._allowed_users is orch._allowed_users

    def test_a_degraded_slack_section_does_not_revoke_anyone(self):
        """Fail-closed means the GRANT survives a torn document, not that it dies.

        A discarded section holds DEFAULTS -- an empty allow-list -- so applying
        it would revoke every guest because the file was briefly malformed.
        """
        from _hot_reload_helpers import cfg_with, change

        from kiro_crew.config.live import ConfigDeferred

        orch = self._orch()
        torn = cfg_with("slack", degraded=frozenset({"slack"}), allowed_users=[])
        with pytest.raises(ConfigDeferred):
            asyncio.run(orch._on_slack_config_change(change(torn, "slack.allowed_users")))
        assert is_guest_user(GUEST) is True

    def test_the_revocation_is_audited(self):
        """A grant changing hands is an authorization event in both directions."""
        from types import SimpleNamespace

        from kiro_crew.slack import gateway as _gw

        audited: list[dict] = []
        orch = self._orch()
        with patch.object(
            _gw,
            "sel",
            lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw)),
        ):
            asyncio.run(orch._on_slack_config_change(self._revoke_to()))
        rows = [a for a in audited if a.get("operation") == "slack.authorization_config_change"]
        assert [r["resources"] for r in rows] == ["slack.allowed_users"]

    def test_no_second_admission_cache_exists_to_reconcile(self):
        """CONTROL-BEARING: the reconcile is complete only if one reader decides.

        Writing the handler global is the whole job precisely because
        ``is_guest_user`` is the only reader that DECIDES anything. The set is
        pinned rather than counted, so a new reader goes red and has to be
        classified here -- a second deciding reader would make this applier a
        half-fix, which is how items 6, 7, B1 and B2 happened.

        ``format_allowlist`` also reads it, and is display-only by its own
        contract: it renders the owner's read-only listing and offers no
        mutation. It is in fact a second thing the missing reconcile broke --
        that listing sourced a revoked guest from this set and kept showing them
        as having access.
        """
        import ast

        src = Path(inspect.getfile(is_guest_user))
        tree = ast.parse(src.read_text(encoding="utf-8"))
        readers = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(isinstance(n, ast.Name) and n.id == "_allowed_users" for n in ast.walk(node))
        }
        # CONTROL: the scan resolves real functions, so an empty-looking result
        # below would be a broken probe rather than a clean module.
        assert "is_guest_user" in readers
        deciders = readers - {"set_allowed_users", "format_allowlist"}
        assert deciders == {"is_guest_user"}, (
            f"a new reader of the admission set appeared: {sorted(deciders)} -- "
            "classify it as deciding (then the applier must feed it) or as display"
        )


# ──────── boundary: a guest's !incognito never reaches the durable flag ────────


class TestGuestIncognitoNeverSetsTheDurableFlag:
    """The privacy chokepoint, asserted at the seam it actually writes.

    ``!incognito`` is not a message-shaped control: it writes a flag into the
    durable ``SessionMap`` that ``hydrate()`` rebuilds on the inbound path, so it
    outlives a restart and suppresses the owner's record of every turn it covers.
    That flag is therefore what these tests read -- NOT
    ``maybe_apply_privacy_modifiers``'s return tuple, which is an intermediate the
    layer below recomputes, and which is exactly the shape of assertion that let
    three defects through a green suite in earlier rounds.

    Read the same way ``privacy_mode.hydrate`` reads it: through the REAL
    ``SessionMap`` the ``SessionManager`` owns. A ``MagicMock`` cannot stand in --
    ``conv_state_map``'s ``isinstance`` check rejects it and the write silently
    goes nowhere, which would make the guest case pass for the wrong reason and
    the owner control fail.
    """

    @staticmethod
    def _sessions(tmp_path):
        smap = _real_index(tmp_path)
        sessions = MagicMock()
        sessions._session_map = smap
        return sessions, smap

    @pytest.fixture(autouse=True)
    def _clear_trackers(self):
        """Drop the process-wide privacy marks this class's keys may leave.

        ``apply_mode`` is idempotent on an already-marked key: it returns early
        and never reaches ``_persist``. A key left marked by an earlier run would
        therefore make the OWNER control observe an unwritten flag and fail,
        which would look like the guard misfiring.
        """
        yield
        for mode in (privacy_mode.MODE_TEMPORARY, privacy_mode.MODE_INCOGNITO):
            tracker = privacy_mode._tracker(mode)
            for key in [k for k in tracker if k.startswith("slack:pm-")]:
                tracker.pop(key, None)

    @pytest.mark.asyncio
    async def test_guest_incognito_leaves_the_flag_unset_and_the_token_in_text(self, tmp_path):
        sessions, smap = self._sessions(tmp_path)
        slack = MagicMock()
        slack.post_message = AsyncMock()
        key = "slack:pm-guest-thread"

        text, cmd_text, only_modifier = await maybe_apply_privacy_modifiers(
            "!incognito",
            "!incognito",
            key,
            GUEST,
            TRACKED,
            slack,
            sessions,
            "10.0",
            guest_user=GUEST,
        )

        # The seam: the durable flag, read as hydrate() reads it.
        assert smap.get_flag(key, privacy_mode.MODE_INCOGNITO) is False
        assert smap.get_flag(canonical_key(key), privacy_mode.MODE_INCOGNITO) is False
        # The token survives, so the guest's own turn answers it as ordinary
        # content rather than the message being swallowed as a command.
        assert "!incognito" in text
        assert "!incognito" in cmd_text
        # False is what keeps the caller going instead of returning early; a
        # modifier-only guest message must still become a turn.
        assert only_modifier is False
        # Nothing was announced to the channel: no mode was applied.
        slack.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_owner_incognito_does_set_the_flag(self, tmp_path):
        """CONTROL. Without this the test above passes on a broken write path."""
        sessions, smap = self._sessions(tmp_path)
        slack = MagicMock()
        slack.post_message = AsyncMock()
        key = "slack:pm-owner-thread"

        text, cmd_text, only_modifier = await maybe_apply_privacy_modifiers(
            "!incognito",
            "!incognito",
            key,
            OWNER,
            TRACKED,
            slack,
            sessions,
            "10.0",
            guest_user="",
        )

        assert smap.get_flag(canonical_key(key), privacy_mode.MODE_INCOGNITO) is True
        assert "!incognito" not in cmd_text
        assert only_modifier is True
        slack.post_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_guest_temporary_leaves_its_flag_unset_too(self, tmp_path):
        """The refusal is the whole function, not the incognito branch of it."""
        sessions, smap = self._sessions(tmp_path)
        slack = MagicMock()
        slack.post_message = AsyncMock()
        key = "slack:pm-guest-temporary"

        text, _cmd, _only = await maybe_apply_privacy_modifiers(
            "!temporary",
            "!temporary",
            key,
            GUEST,
            TRACKED,
            slack,
            sessions,
            "10.0",
            guest_user=GUEST,
        )

        assert smap.get_flag(canonical_key(key), privacy_mode.MODE_TEMPORARY) is False
        assert "!temporary" in text

    def test_the_guard_has_no_default_so_a_new_caller_cannot_forget_it(self):
        """A default of ``""`` would admit a guest by omission at a future site."""
        sig = inspect.signature(maybe_apply_privacy_modifiers)
        param = sig.parameters["guest_user"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty


# ───── boundary: the owner's !stop actually cancels a guest's running turn ─────


class TestOwnerStopCancelsAGuestTurn:
    """The emergency brake, asserted on the task rather than on a resolved key.

    Every routing read hides a guest's thread claim, which is correct. ``!stop``
    is cancelling, not routing, so without the explicit non-hiding read the
    owner's resolution lands on the bare thread key while the live task is keyed
    ``slack:guest-<uid>-<ts>``: the owner is told "Nothing running." and the
    untrusted turn continues.

    So the assertion is that the guest's TASK was cancelled and that the owner was
    not told nothing was running. A test asserting which key the stop resolved to
    would pass against the broken version too, because the broken version also
    resolves a key -- just the wrong one.
    """

    GUEST_THREAD = "900.0"

    def _orch_with_a_live_guest_turn(self, tmp_path, *, claim_is_guest=True):
        """A real index holding a claim, and a real task registered under it."""
        orch = _make_orch()
        smap = _real_index(tmp_path)
        claim_key = (
            guest_session_key(GUEST, self.GUEST_THREAD)
            if claim_is_guest
            else canonical_key(self.GUEST_THREAD)
        )
        smap.set_slack_link(claim_key, self.GUEST_THREAD, TRACKED)

        # The real index rule decides what the brake can see, not a stub return.
        orch.sessions.guest_claim_for_thread = smap.guest_claim_for_thread
        # The hiding rule, which is the condition the bug lived under: the owner's
        # routing read sees nothing in this thread.
        orch.sessions.get_session_for_thread = MagicMock(return_value=None)
        orch.sessions.has_session = MagicMock(side_effect=lambda k: k == claim_key)
        orch.sessions.stop_turn = AsyncMock(
            side_effect=lambda k, **kw: "soft" if k == claim_key else "idle"
        )
        orch.sessions.note_stop = MagicMock()
        orch.slack.post_message = AsyncMock()

        async def _running_turn():
            await asyncio.sleep(3600)

        task = asyncio.get_running_loop().create_task(_running_turn())
        orch._session_tasks[claim_key] = task
        return orch, claim_key, task

    @staticmethod
    def _said_nothing_running(orch) -> bool:
        return any(
            "Nothing running." in str(c.args) for c in orch.slack.post_message.await_args_list
        )

    @pytest.mark.asyncio
    async def test_owner_stop_cancels_the_guest_task(self, tmp_path):
        orch, claim_key, task = self._orch_with_a_live_guest_turn(tmp_path)

        await _route(
            orch,
            user=OWNER,
            text="!stop",
            ts="901.0",
            thread=self.GUEST_THREAD,
        )
        await asyncio.sleep(0)

        # The seam: the guest's turn is dead.
        assert task.cancelled() or task.done()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert task.cancelled()
        # And the owner was not told the brake found nothing.
        assert not self._said_nothing_running(orch)
        # The stop reached the guest's own key, popped from the registry so a
        # later stop cannot re-cancel a dead task.
        assert claim_key not in orch._session_tasks
        assert claim_key in [c.args[0] for c in orch.sessions.stop_turn.await_args_list]

    @pytest.mark.asyncio
    async def test_without_the_guest_claim_the_same_drive_finds_nothing(self, tmp_path):
        """CONTROL. Proves the claim lookup is what does the work above.

        Same orchestrator, same message, but the thread is claimed by an ORDINARY
        session -- which ``guest_claim_for_thread`` returns None for, because that
        claim is already visible through the routing read. The registered task is
        then unreachable from the bare thread key and survives, which is precisely
        the failure the fix removes for the guest case.
        """
        orch, claim_key, task = self._orch_with_a_live_guest_turn(tmp_path, claim_is_guest=False)
        assert orch.sessions.guest_claim_for_thread(self.GUEST_THREAD) is None

        await _route(
            orch,
            user=OWNER,
            text="!stop",
            ts="901.0",
            thread=self.GUEST_THREAD,
        )
        await asyncio.sleep(0)

        assert not task.cancelled()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_the_brake_stops_the_owners_own_key_as_well(self, tmp_path):
        """Both keys, not one: the owner's turns in that thread stay stoppable.

        Replacing the resolved key with the guest's would trade one unstoppable
        turn for another, so the stop must carry both.
        """
        orch, claim_key, task = self._orch_with_a_live_guest_turn(tmp_path)

        await _route(
            orch,
            user=OWNER,
            text="!stop",
            ts="901.0",
            thread=self.GUEST_THREAD,
        )
        await asyncio.sleep(0)
        with contextlib.suppress(asyncio.CancelledError):
            await task

        stopped = [c.args[0] for c in orch.sessions.stop_turn.await_args_list]
        assert claim_key in stopped
        # The owner's own key is the BARE thread_ts here: the stop block resolves
        # ``session_key = _flat_stop_key or (thread_ts or msg_ts)`` and leaves
        # canonicalization to ``stop_turn``, so asserting the canonical spelling
        # would be asserting a transformation a lower layer performs.
        assert self.GUEST_THREAD in stopped
        assert len(stopped) == 2

    @pytest.mark.asyncio
    async def test_a_session_double_without_the_method_does_not_break_the_brake(self, tmp_path):
        """The brake must survive a ``sessions`` object that has no such method.

        ``orch.sessions`` is typed loosely and the focused doubles in the channel
        suites predate this method, so the read is probed and type-narrowed. Both
        failure shapes are covered: an object missing the attribute entirely, and
        one whose stand-in returns an auto-attribute mock. Either becoming a key
        raises ``TypeError`` inside the owner's emergency brake -- every consumer
        treats these as strings, including the ``","``-join into the audit record.
        """
        for _label, _read in (
            ("missing", None),
            ("mock", MagicMock(return_value=MagicMock())),
        ):
            orch, claim_key, task = self._orch_with_a_live_guest_turn(tmp_path)
            if _read is None:
                del orch.sessions.guest_claim_for_thread
            else:
                orch.sessions.guest_claim_for_thread = _read
            # Its own key still has a live task, so the brake still does its job.
            orch.sessions.has_session = MagicMock(side_effect=lambda k: k == self.GUEST_THREAD)
            orch.sessions.stop_turn = AsyncMock(return_value="soft")
            orch._session_tasks[self.GUEST_THREAD] = orch._session_tasks.pop(claim_key)
            task_for_owner = orch._session_tasks[self.GUEST_THREAD]

            await _route(
                orch,
                user=OWNER,
                text="!stop",
                ts="901.0",
                thread=self.GUEST_THREAD,
            )
            await asyncio.sleep(0)
            with contextlib.suppress(asyncio.CancelledError):
                await task_for_owner

            stopped = [c.args[0] for c in orch.sessions.stop_turn.await_args_list]
            assert stopped == [self.GUEST_THREAD], _label
            assert all(isinstance(k, str) for k in stopped), _label

    def test_the_bypass_is_not_reachable_from_a_routing_read(self):
        """The brake's read is a method of its own, not a flag on the router.

        A boolean on ``get_session_for_thread`` would be copied by a later caller
        who wanted "the session for this thread" and would re-open the adoption
        hole the hiding rule closes.
        """
        from kiro_crew.session_map import SessionMap

        sig = inspect.signature(SessionMap.get_session_for_thread)
        assert "guest_user" in sig.parameters
        assert not any(
            p for name, p in sig.parameters.items() if name in {"reveal_guest", "include_guest"}
        )
        doc = inspect.getdoc(SessionMap.guest_claim_for_thread) or ""
        assert "CANCELLATION" in doc


# ───── the two guest specs are separate files, and stay separate ──────────────


class TestTheTwoGuestSpecsDoNotCollide:
    """``kirocrew-guest`` is not ``kirocrew-slack-guest``, and neither writes the other.

    This is the defect that produced four red lanes: both installers wrote the
    same filename, Python bound the later ``def``, and every existing caller then
    emitted the Slack spec. So ``messaging.dispatch``'s channel-wide tool-less
    boundary -- ``tools: []``, pinned by its own test in
    ``test_messaging_dispatch.py`` -- was never written at all, and its prompt,
    which tells the model it has no tools, became unreachable while the spec
    mounted one.

    Asserted on the FILES both installers write, after running both in one
    process, which is the condition the collision needed.
    """

    @staticmethod
    def _install_both(tmp_path):
        from kiro_crew import agent as agent_mod

        with patch.object(agent_mod, "kiro_agents_dir_path", lambda: tmp_path):
            agent_mod._install_guest_agent()
            agent_mod._install_slack_guest_agent()

    def test_the_two_filenames_are_different(self):
        from kiro_crew.agent_files import GUEST_AGENT_FILENAME

        assert GUEST_AGENT_FILENAME != SLACK_GUEST_AGENT_FILENAME

    def test_the_channel_wide_boundary_stays_tool_less(self, tmp_path):
        import json

        from kiro_crew.agent_files import GUEST_AGENT_FILENAME

        self._install_both(tmp_path)
        spec = json.loads((tmp_path / GUEST_AGENT_FILENAME).read_text(encoding="utf-8"))
        # The seam: what the tool-less boundary actually mounts, after the Slack
        # installer has also run.
        assert spec["tools"] == []
        assert spec["mcpServers"] == {}
        assert spec["name"] == "kirocrew-guest"

    def test_the_slack_spec_keeps_its_own_tool(self, tmp_path):
        import json

        self._install_both(tmp_path)
        spec = json.loads((tmp_path / SLACK_GUEST_AGENT_FILENAME).read_text(encoding="utf-8"))
        assert spec["tools"] == ["web_search"]
        assert spec["name"] == SLACK_GUEST_AGENT_NAME

    def test_both_files_exist_so_neither_installer_overwrote_the_other(self, tmp_path):
        from kiro_crew.agent_files import GUEST_AGENT_FILENAME

        self._install_both(tmp_path)
        assert (tmp_path / GUEST_AGENT_FILENAME).is_file()
        assert (tmp_path / SLACK_GUEST_AGENT_FILENAME).is_file()

    def test_the_toolless_turn_agent_names_the_channel_wide_spec(self):
        """The cross-channel boundary must not resolve to the Slack guest spec.

        ``TOOLLESS_TURN_AGENT`` is what every channel's non-operator turn is driven
        on, so pointing it at a spec that mounts a tool would widen a boundary this
        PR has no business touching.
        """
        from kiro_crew.agent_files import GUEST_AGENT_FILENAME
        from kiro_crew.messaging.dispatch import TOOLLESS_TURN_AGENT

        assert TOOLLESS_TURN_AGENT == GUEST_AGENT_FILENAME.removesuffix(".json")
        assert TOOLLESS_TURN_AGENT != SLACK_GUEST_AGENT_NAME

    def test_both_specs_are_withheld_from_the_chat_picker(self):
        """Neither is a sensible template to run a chat AS, for different reasons."""
        from kiro_crew.agent_files import GUEST_AGENT_FILENAME
        from kiro_crew.dashboard.handlers.agent_catalog import _BACKGROUND_ONLY_FILES

        assert GUEST_AGENT_FILENAME in _BACKGROUND_ONLY_FILES
        assert SLACK_GUEST_AGENT_FILENAME in _BACKGROUND_ONLY_FILES

    def test_the_new_spec_is_labelled_kirocrew_owned(self):
        """The picker's hiding rule requires the owned label, so the rename needs it.

        ``_is_background_only`` is ``kirocrew_owned and filename in
        _BACKGROUND_ONLY_FILES``: without the discovery label the entry above would
        be inert and the spec would be offered as a chat template.
        """
        src = Path(inspect.getfile(guest_member_configured)).parent.parent
        discovery = (src / "agent_discovery.py").read_text(encoding="utf-8")
        assert "SLACK_GUEST_AGENT_FILENAME" in discovery


# ──── a backend that cannot honour the spec refuses the guest at admission ─────


class TestGuestRefusedWhenTheBackendCannotGateTools:
    """Reuses ``messaging.dispatch.toolless_turns_supported`` rather than re-deriving.

    The guest's whole tool posture assumes the harness mounts what the spec names,
    which is true only on ``Routing.AGENT_SPEC``. Elsewhere the spec decides
    nothing: a harness reading no agent spec keeps its native tools, and a
    project-preapproved one raises no permission request -- and the permission
    request is the only place the guest tool gate runs.

    The repo had already answered this question for its other non-operator path,
    which is why this is the shipped predicate and not a new check.
    """

    @pytest.mark.asyncio
    async def test_a_guest_is_refused_on_a_backend_that_reads_no_agent_spec(self):
        orch = _make_orch()
        cfg = _cfg()
        cfg.agent.acp_backend = "claude_code"
        # Precondition, so a pass cannot come from the backend being servable.
        from kiro_crew.messaging.dispatch import toolless_turns_supported

        assert toolless_turns_supported("claude_code") is False

        hm = await _route(orch, loaded_cfg=cfg)

        # The seam: no turn ran.
        hm.assert_not_called()
        text = "".join(
            str(c.kwargs.get("text", "")) + str(c.args)
            for c in orch.slack.post_ephemeral.await_args_list
        )
        assert "cannot limit a guest's tools" in text
        assert "add you to the allowlist" not in text

    @pytest.mark.asyncio
    async def test_the_same_guest_is_admitted_on_the_spec_mounting_backend(self):
        """CONTROL. Without it the refusal above could be any other denial.

        ``ACP_BACKEND_KIRO`` is ``""``, i.e. the DEFAULT when nothing is
        configured, so a default install is unaffected by the refusal above --
        named through the constant rather than spelled, so the control cannot
        drift from the routing table.
        """
        from kiro_crew.agent_sdk.backends import ACP_BACKEND_KIRO
        from kiro_crew.messaging.dispatch import toolless_turns_supported

        orch = _make_orch()
        cfg = _cfg()
        cfg.agent.acp_backend = ACP_BACKEND_KIRO
        assert toolless_turns_supported(ACP_BACKEND_KIRO) is True

        hm = await _route(orch, loaded_cfg=cfg)

        hm.assert_called_once()
        assert hm.await_args.kwargs.get("guest_user") == GUEST

    def test_admission_reads_the_shipped_predicate_and_not_its_own_copy(self):
        """A second copy of the routing rule is how the two paths drift apart.

        Checked on the IMPORTS, not on the source text: the comment at the refusal
        names ``Routing.AGENT_SPEC`` to explain why the predicate is the right one,
        and a substring scan would read that prose as a re-derivation.
        """
        import ast

        tree = ast.parse(Path(inspect.getfile(_route_message)).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        # CONTROL: the scan resolves real imports.
        assert "toolless_turns_supported" in imported
        # The Slack layer asks the question; it does not own the answer.
        assert "Routing" not in imported
        assert "routing_for" not in imported

    def test_slack_warns_at_start_like_the_other_adopter(self):
        """The owner learns from a log line, not from the first guest's refusal."""
        from kiro_crew.slack import gateway as gw

        assert hasattr(gw, "_warn_if_guest_turns_unservable")
        src = inspect.getsource(gw._warn_if_guest_turns_unservable)
        assert "warn_if_toolless_turns_unservable" in src
        # It supplies the admission facts only; the decision stays on the seam.
        assert "ACP_BACKEND_ROUTING" not in src


# ───── the guest turn's working directory: already isolated, already cold ──────


class TestTheGuestCwdIsAlreadyPerGuestAndCold:
    """Recorded rather than re-mechanised: this property is already true.

    A session's working directory is ``workspace_root() / _safe_dir_name(key)``
    (``config.loader._session_work_dir``). A guest's key is
    ``slack:guest-<uid>-<ts>``, unique per guest AND per thread, so the directory
    is the guest's own and is new the first time that key is used. No second
    mechanism is added for it; these assertions exist so the property cannot
    regress silently if the key shape changes.

    What this is NOT: an OS boundary. The directory is a sibling under the same
    workspace root, so it bounds where the turn STARTS, not what it could reach
    with a filesystem tool -- and none is approved.
    """

    @staticmethod
    def _dir_for(key, tmp_path):
        from kiro_crew.config import loader as loader_mod

        with patch.object(loader_mod, "workspace_root", lambda create=True: tmp_path):
            return loader_mod._session_work_dir(key)

    def test_a_guest_key_does_not_share_the_owners_directory(self, tmp_path):
        guest = self._dir_for(guest_session_key(GUEST, "10.0"), tmp_path)
        owner = self._dir_for(canonical_key("10.0"), tmp_path)
        assert guest != owner

    def test_two_guests_in_the_same_thread_do_not_share_one(self, tmp_path):
        a = self._dir_for(guest_session_key(GUEST, "10.0"), tmp_path)
        b = self._dir_for(guest_session_key("U0OTHER01", "10.0"), tmp_path)
        assert a != b

    def test_a_new_thread_is_a_new_directory(self, tmp_path):
        first = self._dir_for(guest_session_key(GUEST, "10.0"), tmp_path)
        second = self._dir_for(guest_session_key(GUEST, "11.0"), tmp_path)
        assert first != second
        # Cold: neither exists until something creates it.
        assert not first.exists()
        assert not second.exists()

    def test_the_directory_name_carries_the_guest_identity(self, tmp_path):
        d = self._dir_for(guest_session_key(GUEST, "10.0"), tmp_path)
        assert GUEST in d.name
        assert "guest-" in d.name
