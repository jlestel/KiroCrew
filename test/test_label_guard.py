"""The shared label guard: one refusal/prose check for every label path.

#10375: ``_looks_like_prose`` lived only in ``dashboard/chat_title.py`` and was
called from exactly one place, so the Slack/Telegram namer, the nav link chips
and the session summary each stored a model refusal. The behavioural tests per
path live beside those paths; this file pins the two things only the module
can promise: the ceilings are parameters, and the dashboard title still runs
THIS code rather than a private copy that could drift again.
"""

from __future__ import annotations

from kiro_crew import label_guard
from kiro_crew.dashboard import chat_title
from kiro_crew.label_guard import is_verdict_reply, looks_like_prose


def test_dashboard_title_delegates_to_the_shared_guard():
    """Mutation: give chat_title back a private ``_looks_like_prose`` -- red."""
    assert chat_title._looks_like_prose is looks_like_prose
    assert chat_title._is_verdict_reply is is_verdict_reply


def test_word_ceiling_is_a_parameter():
    """A 15-word line is prose for a 3-6 word title but a legitimate 18-word
    summary. Mutation: ignore ``max_words`` -- red on one of the two."""
    fifteen = " ".join(["word"] * 15)
    assert looks_like_prose(fifteen)
    assert not looks_like_prose(fifteen, max_words=36)


def test_unspaced_ceiling_is_a_parameter():
    """Same for the unspaced-script ceiling (CJK is one ``str.split`` word)."""
    thirty = "修" * 30
    assert looks_like_prose(thirty)
    assert not looks_like_prose(thirty, max_unspaced_chars=72)


def test_openers_fire_regardless_of_ceilings():
    """A refusal is caught by shape even when the ceilings are generous."""
    assert looks_like_prose("I cannot access that link", max_words=1000, max_unspaced_chars=1000)
    assert looks_like_prose("\uc8c4\uc1a1\ud569\ub2c8\ub2e4", max_words=1000)


def test_verdict_with_reason_versus_title_opening_with_the_word():
    assert is_verdict_reply("SKIP - too vague", ("SKIP",))
    assert is_verdict_reply("skip: greetings only", ("SKIP",))
    assert not is_verdict_reply("SKIP and KEEP handling", ("SKIP",))
    assert not is_verdict_reply("SKIP_TESTS env var flag", ("SKIP",))


def test_title_defaults_match_the_documented_contract():
    assert label_guard.TITLE_MAX_WORDS == 12
    assert label_guard.TITLE_MAX_UNSPACED_CHARS == 24
