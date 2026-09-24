"""What the writer does with an entry it cannot simply upload.

The contract this subsystem is held to has one rule above the others: nothing in the
backup set is skipped quietly. So every branch that could end in "this object was not
uploaded" is pinned here, and each is pinned on the OBSERVABLE the operator gets --
the cycle result, and whether the cycle raises -- rather than on a log line.

The set splits three ways and the difference matters:

* **Too large.** Uploaded anyway. The restore side will refuse to read it, so the pair
  is incomplete for that one conversation, and the cycle says so.
* **Not a file whose bytes can be uploaded.** Refused, and the cycle raises. A link or
  a FIFO where a transcript belongs is not a transcript.
* **Gone.** Not a failure. An entry listed and then removed is a conversation its owner
  deleted between the two steps.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from container.common import keys
from container.sidecar import backup as backup_mod
from container.sidecar.store import ObjectAbsent

from ._settings_helper import make_settings

STEM = "dashboard_cust-77"


class _RecordingStore:
    """Accepts every put and remembers the keys and byte counts.

    ``get`` answers ABSENT rather than raising ``KeyError``, because that is the real
    store's contract and the cycle reads the two differently: an absent generation pointer
    is a first boot, while a pointer that could not be read withholds the authority pair
    for its own reason. A fake that raised ``KeyError`` withheld the pair in every test
    here, which hid whichever branch the test meant to measure.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put(
        self,
        key: str,
        body,
        size: int,
        *,
        budget: float | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> None:
        self.objects[key] = body.read(size)

    def get(self, key: str, *, limit: int) -> bytes:
        try:
            return self.objects[key]
        except KeyError:
            raise ObjectAbsent(key) from None


class _FailingStore:
    """Fails the put for one key and accepts the rest."""

    def __init__(self, failing_suffix: str) -> None:
        self.failing_suffix = failing_suffix
        self.objects: dict[str, bytes] = {}

    def put(
        self,
        key: str,
        body,
        size: int,
        *,
        budget: float | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> None:
        if key.endswith(self.failing_suffix):
            raise RuntimeError("SlowDown")
        self.objects[key] = body.read(size)

    def get(self, key: str, *, limit: int) -> bytes:
        try:
            return self.objects[key]
        except KeyError:
            raise ObjectAbsent(key) from None


def _settings(tmp_path: Path):
    s = make_settings(tmp_path, crew="crew-77", prefix="crews")
    for name in keys.AUTHORITY_NAMES:
        (s.config_dir / name).write_bytes(b"{}")
    return s


def _transcript(settings, payload: bytes, stem: str = STEM) -> Path:
    path = settings.sessions_dir / f"{stem}{keys.TRANSCRIPT_SUFFIX}"
    path.write_bytes(payload)
    return path


def test_an_object_above_the_ceiling_is_uploaded_and_named(tmp_path, monkeypatch):
    """The size that makes it unreadable on the way back does not make it unwritten."""
    settings = _settings(tmp_path)
    payload = b"x" * 64
    _transcript(settings, payload)
    monkeypatch.setattr(backup_mod, "MAX_OBJECT_BYTES", 8)
    store = _RecordingStore()

    result = backup_mod.run_cycle(settings, store, state={})

    key = keys.transcript_key(settings, STEM)
    assert result.above_ceiling == [key]
    assert key in result.uploaded
    assert store.objects[key] == payload
    assert result.complete


def test_a_symlink_where_a_transcript_belongs_is_refused_loudly(tmp_path):
    settings = _settings(tmp_path)
    real = tmp_path / "outside.jsonl"
    real.write_bytes(b"not this task's bytes\n")
    link = settings.sessions_dir / f"{STEM}{keys.TRANSCRIPT_SUFFIX}"
    link.symlink_to(real)
    store = _RecordingStore()

    with pytest.raises(backup_mod.BackupIncomplete) as caught:
        backup_mod.run_cycle(settings, store, state={})

    # Unreachable rather than refused: the link is the entry's shape, so every later
    # cycle meets it too and no wait can end.
    assert [name for name, _ in caught.value.result.unreachable] == [link.name]
    assert caught.value.result.refused == []
    assert keys.transcript_key(settings, STEM) not in store.objects


def test_a_fifo_where_a_transcript_belongs_is_refused_without_hanging(tmp_path):
    """A FIFO opened for reading blocks until a writer arrives, so it is refused."""
    settings = _settings(tmp_path)
    path = settings.sessions_dir / f"{STEM}{keys.TRANSCRIPT_SUFFIX}"
    os.mkfifo(path)

    with pytest.raises(backup_mod.BackupIncomplete) as caught:
        backup_mod.run_cycle(settings, _RecordingStore(), state={})

    assert [name for name, _ in caught.value.result.unreachable] == [path.name]
    assert caught.value.result.refused == []


def test_an_entry_removed_after_it_is_listed_is_not_a_failure(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    path = _transcript(settings, b"turn\n")
    real_open = backup_mod.open_snapshot

    def _open_after_deleting(target: Path, *, root: Path):
        if target == path:
            path.unlink()
        return real_open(target, root=root)

    monkeypatch.setattr(backup_mod, "open_snapshot", _open_after_deleting)

    result = backup_mod.run_cycle(settings, _RecordingStore(), state={})

    assert result.gone == [path.name]
    assert result.complete


def test_a_failed_upload_raises_and_is_not_recorded_as_done(tmp_path):
    """The next cycle must resend it, so the fingerprint map keeps only successes."""
    settings = _settings(tmp_path)
    _transcript(settings, b"turn\n")
    state: dict = {}
    key = keys.transcript_key(settings, STEM)

    with pytest.raises(backup_mod.BackupIncomplete):
        backup_mod.run_cycle(settings, _FailingStore(f"{STEM}.jsonl"), state=state)

    assert key not in state
    store = _RecordingStore()
    second = backup_mod.run_cycle(settings, store, state=state)
    assert key in second.uploaded
    assert store.objects[key] == b"turn\n"


def test_a_failed_upload_withholds_the_authority_files(tmp_path):
    """A partial cycle must not advance the index past the bytes it failed to write.

    The authority files name the transcripts, so publishing them here would point a
    replacement at an object that is not in the bucket. Leaving them alone keeps the
    pair at the last cycle that completed: older, and true.

    The failure is an UPLOAD failure, which is the case withholding is for: the next
    cycle retries the object and publishes a pair the bucket supports. Planting a FIFO
    instead looks like the same thing and is not -- see the two tests below.
    """
    settings = _settings(tmp_path)
    _transcript(settings, b"turn\n")
    store = _FailingStore(f"{STEM}{keys.TRANSCRIPT_SUFFIX}")

    with pytest.raises(backup_mod.BackupIncomplete) as caught:
        backup_mod.run_cycle(settings, store, state={})

    assert [name for name, _why in caught.value.result.refused] == [
        f"{STEM}{keys.TRANSCRIPT_SUFFIX}"
    ]
    for name in keys.AUTHORITY_NAMES:
        assert keys.authority_slot_key(settings, keys.AUTHORITY_SLOTS[0], name) not in store.objects
    assert set(caught.value.result.withheld) == {
        keys.authority_slot_key(settings, keys.AUTHORITY_SLOTS[0], name)
        for name in keys.AUTHORITY_NAMES
    }


def test_an_unreachable_entry_does_not_withhold_the_authority_files(tmp_path):
    """Withholding is a WAIT, and a planted name is not something to wait for.

    A FIFO or a symlink at a transcript's path is refused by every cycle, not just this
    one, so withholding the pair on it withholds the pair FOREVER. The index then freezes
    at the moment the entry appeared while transcripts keep uploading past it, and the
    next task replacement restores a conversation list that predates every conversation
    served since -- an unbounded, silent loss, traded for a bounded one.

    So the pair is published. The cycle is still incomplete and the entry is still named,
    which is what an operator needs to remove it.
    """
    settings = _settings(tmp_path)
    _transcript(settings, b"turn\n")
    os.mkfifo(settings.sessions_dir / f"planted{keys.TRANSCRIPT_SUFFIX}")
    store = _RecordingStore()

    with pytest.raises(backup_mod.BackupIncomplete) as caught:
        backup_mod.run_cycle(settings, store, state={})

    for name in keys.AUTHORITY_NAMES:
        assert keys.authority_slot_key(settings, keys.AUTHORITY_SLOTS[0], name) in store.objects
    assert caught.value.result.withheld == []
    assert (
        not caught.value.result.complete
    ), "the cycle did not reach everything, so it must not report a lossless one"


def test_a_second_cycle_publishes_the_pair_again_with_the_entry_still_there(tmp_path):
    """The property the old withholding destroyed: the index keeps up with the bucket.

    One cycle publishing the pair is not the claim -- a freeze shows itself on the cycle
    AFTER the entry appears. So a second cycle runs with the same planted name and a new
    conversation, and the pair it publishes has to reflect that conversation rather than
    staying where the first cycle left it.
    """
    settings = _settings(tmp_path)
    _transcript(settings, b"turn\n")
    os.mkfifo(settings.sessions_dir / f"planted{keys.TRANSCRIPT_SUFFIX}")
    state: dict = {}
    store = _RecordingStore()

    with pytest.raises(backup_mod.BackupIncomplete):
        backup_mod.run_cycle(settings, store, state=state)

    (settings.config_dir / "session_map.json").write_bytes(b'{"slot-2": "sid-2"}')
    _transcript(settings, b"a later turn\n", stem="dashboard_cust-92")

    with pytest.raises(backup_mod.BackupIncomplete):
        backup_mod.run_cycle(settings, store, state=state)

    # Slot-agnostic on purpose: the generation protocol alternates slots, so naming one
    # would pin the alternation instead of the property. These bytes reached the config
    # file only between the two cycles, so only the second cycle can have published them.
    published = [raw for key, raw in store.objects.items() if key.endswith("session_map.json")]
    assert b'{"slot-2": "sid-2"}' in published, (
        "the second cycle must republish the index; a pair frozen at the first cycle is "
        "the permanent freeze this split exists to prevent"
    )
    assert keys.transcript_key(settings, "dashboard_cust-92") in store.objects


def test_a_directory_named_like_a_transcript_is_not_in_the_set(tmp_path):
    settings = _settings(tmp_path)
    (settings.sessions_dir / f"{STEM}{keys.TRANSCRIPT_SUFFIX}").mkdir()

    names = [key for key, _ in backup_mod.objects_to_back_up(settings).data]

    assert keys.transcript_key(settings, STEM) not in names


def test_a_file_without_the_transcript_suffix_is_not_in_the_set(tmp_path):
    settings = _settings(tmp_path)
    (settings.sessions_dir / "notes.txt").write_bytes(b"scratch\n")

    paths = [path.name for _, path in backup_mod.objects_to_back_up(settings).data]

    assert "notes.txt" not in paths
