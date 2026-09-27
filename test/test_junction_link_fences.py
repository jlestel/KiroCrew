"""A directory link fence must see a Windows JUNCTION, not only a symlink.

``os.path.islink`` and ``Path.is_symlink()`` both answer False for a Windows
directory junction -- a reparse point that ``is_dir()`` answers True for and
that ``os.walk`` descends through even with ``followlinks=False``, as does
``Path.rglob``. A fence written with either predicate therefore fails OPEN for
the one link type an unprivileged Windows user can actually create, while
reading green on POSIX where the same plant is a symlink.

``platform_compat.is_link_or_junction`` is the repository's answer and is what
``docs/system-specs/common/platform-compat.md`` mandates for "detect/remove a
dir link". These pin the five call sites that were still testing the narrower
predicate, each by the consequence that made it worth fixing rather than by the
predicate itself -- a later refactor that keeps the behaviour is free to move.

Links are staged through ``conftest.make_dir_link``: a plain symlink on POSIX, a
real junction on Windows. So every test runs on every host, and the junction arm
-- the one the defect lived in -- is exercised where it exists. The negative
control in :class:`TestThePlantIsReallyAJunctionOnWindows` is what keeps that
true; without it a host that quietly produced a symlink would keep reporting
green while testing nothing.
"""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path
from unittest import mock

import pytest

from conftest import make_dir_link
from kiro_crew import artifacts, node_modules_txn, platform_compat, portability, skills


def _seed_victim(root: Path) -> Path:
    """A directory OUTSIDE the tree under test, holding a file nothing may touch."""
    victim = root / "victim"
    victim.mkdir()
    (victim / "precious.txt").write_text("not the caller's to delete", encoding="utf-8")
    return victim


class TestThePlantIsReallyAJunctionOnWindows:
    """Guards the guard.

    Every test below stages its link with ``make_dir_link``. If that produced
    something ``Path.is_symlink()`` already catches on this host, the fences
    would pass for the old reason and the tests would cease to be evidence
    while still reporting green.
    """

    def test_the_link_is_a_reparse_point_the_narrow_predicate_misses(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        make_dir_link(link, target)
        assert platform_compat.is_link_or_junction(link)
        if os.name == "nt":
            assert not link.is_symlink(), "expected a junction -- the arm under test"
            assert not os.path.islink(link), "expected a junction -- the arm under test"

    def test_both_walkers_descend_through_it(self, tmp_path: Path) -> None:
        """Why pruning, not flagging, is the fix in the two delete paths below.

        A walk that reaches a link's target enumerates files the caller then
        judges as its own, so the fence has to stop the ENUMERATION.
        """
        victim = _seed_victim(tmp_path)
        tree = tmp_path / "tree"
        tree.mkdir()
        make_dir_link(tree / "link", victim)

        walked = [
            os.path.relpath(os.path.join(dirpath, name), tree)
            for dirpath, _dirs, files in os.walk(tree)
            for name in files
        ]
        rglobbed = [p.name for p in tree.rglob("*") if p.is_file()]
        if os.name == "nt":
            assert walked == [
                os.path.join("link", "precious.txt")
            ], "os.walk descends a junction even with followlinks=False"
            assert rglobbed == ["precious.txt"], "rglob descends a junction"
        else:
            assert walked == [], "os.walk does not follow a POSIX symlink"


class TestArtifactTrashDoesNotDeleteThroughALink:
    """``ArtifactStore._rmtree`` is a hand-rolled recursive delete.

    It enumerated with ``rglob`` deepest-first and unlinked anything
    ``is_file()``. Through a junction that meant the LINK TARGET's files were
    unlinked first -- data loss outside the artifact store, reported as a
    successful prune.
    """

    def test_a_linked_directory_is_removed_without_its_target(self, tmp_path: Path) -> None:
        victim = _seed_victim(tmp_path)
        doomed = tmp_path / "doomed"
        (doomed / "nested").mkdir(parents=True)
        (doomed / "nested" / "own.txt").write_text("mine", encoding="utf-8")
        make_dir_link(doomed / "link", victim)

        artifacts.ArtifactStore._rmtree(doomed)

        assert not doomed.exists(), "the store's own tree is still removed"
        assert (victim / "precious.txt").read_text(encoding="utf-8") == (
            "not the caller's to delete"
        ), "a file behind the link was deleted"
        assert victim.is_dir(), "the link target itself was removed"

    def test_a_root_that_survives_propagates_instead_of_warning(self, tmp_path: Path) -> None:
        """The removal must not swallow the ROOT's failure.

        The caller logs a successful delete and fires its event unconditionally, so
        a root left on disk that only warned would report a removal that did not
        happen.
        """
        doomed = tmp_path / "doomed"
        doomed.mkdir()
        (doomed / "held.txt").write_text("a locked store file", encoding="utf-8")

        real_unlink = os.unlink

        def refuse_that_one(target, *a, **kw):  # type: ignore[no-untyped-def]
            if os.path.basename(os.fspath(target)) == "held.txt":
                raise PermissionError(32, "The process cannot access the file")
            return real_unlink(target, *a, **kw)

        with mock.patch.object(artifacts.os, "unlink", refuse_that_one):
            with pytest.raises(OSError):
                artifacts.ArtifactStore._rmtree(doomed)

        assert (doomed / "held.txt").exists(), "the fixture did not hold the file"

    def test_a_nested_residual_is_named_and_still_fails_the_root(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A residual under a SUBdirectory keeps the root non-empty, so the root
        removal fails too.

        The per-entry warnings are diagnostics that name WHERE the residual is; they
        are not a decision to report success. Anything else would be a contradiction:
        a nested file that will not go cannot leave an empty root.
        """
        doomed = tmp_path / "doomed"
        (doomed / "nested").mkdir(parents=True)
        (doomed / "nested" / "held.txt").write_text("locked", encoding="utf-8")

        real_unlink = os.unlink

        def refuse_that_one(target, *a, **kw):  # type: ignore[no-untyped-def]
            if os.path.basename(os.fspath(target)) == "held.txt":
                raise PermissionError(32, "The process cannot access the file")
            return real_unlink(target, *a, **kw)

        with caplog.at_level("WARNING"):
            with mock.patch.object(artifacts.os, "unlink", refuse_that_one):
                with pytest.raises(OSError):
                    artifacts.ArtifactStore._rmtree(doomed)

        assert (doomed / "nested" / "held.txt").exists()
        assert any(
            "held.txt" in r.getMessage() for r in caplog.records
        ), "the residual was not named in a warning"

    def test_a_reparse_point_swapped_in_mid_sweep_is_not_descended(self, tmp_path: Path) -> None:
        """The race the pin closes.

        A child that screened clean is turned into a link before the sweep reaches
        it. ``os.walk`` would descend that (its own re-check is ``os.path.islink``,
        False for a junction) and the target's files would be unlinked. The pinned
        open refuses a link at the name instead, so the link is removed and what it
        points at is untouched.

        The swap is driven from inside ``PinnedDirectory.child`` rather than from
        ``pin_directory``, because only the Windows branch of ``child`` goes through
        that function -- patching it would make this test a no-op on POSIX while
        still reporting green.
        """
        victim = _seed_victim(tmp_path)
        doomed = tmp_path / "doomed"
        swapme = doomed / "swapme"
        swapme.mkdir(parents=True)
        (swapme / "own.txt").write_text("mine", encoding="utf-8")

        real_child = platform_compat.PinnedDirectory.child
        swapped: list[str] = []

        def swap_then_open(self, name):  # type: ignore[no-untyped-def]
            # Stands in for the agent replacing the child between the screen and
            # this open. The real open then runs against what is actually there.
            if name == "swapme" and not swapped:
                swapped.append(name)
                target = Path(self.path) / name
                for leftover in target.iterdir():
                    leftover.unlink()
                target.rmdir()
                make_dir_link(target, victim)
            return real_child(self, name)

        with mock.patch.object(platform_compat.PinnedDirectory, "child", swap_then_open):
            artifacts.ArtifactStore._rmtree(doomed)

        assert swapped, "the fixture never swapped the child, so nothing was proved"
        assert not doomed.exists(), "the store's own tree is still removed"
        assert (victim / "precious.txt").read_text(encoding="utf-8") == (
            "not the caller's to delete"
        ), "the swapped-in link was descended and its target was deleted"
        assert victim.is_dir(), "the link target itself was removed"

    @pytest.mark.parametrize("refusal_errno", [errno.ENOTDIR, errno.ELOOP, errno.EPERM])
    def test_the_swap_is_handled_by_what_is_at_the_name_not_by_the_errno(
        self, tmp_path: Path, refusal_errno: int
    ) -> None:
        """Which error a refused pin reports is not a stable fact, so it is not read.

        ``pin_directory`` says a POSIX symlink fails with "whichever of ENOTDIR /
        ELOOP the kernel reports" for ``O_DIRECTORY | O_NOFOLLOW``; Linux answers
        ENOTDIR and the Windows open raises ``NotADirectoryError`` for a reparse
        point. A handler keyed on one class leaves the link behind wherever the
        kernel picks another, and the root removal then fails with ENOTEMPTY out of
        the caller. Each errno here stands for one of those kernels.
        """
        victim = _seed_victim(tmp_path)
        doomed = tmp_path / "doomed"
        swapme = doomed / "swapme"
        swapme.mkdir(parents=True)
        (swapme / "own.txt").write_text("mine", encoding="utf-8")

        real_child = platform_compat.PinnedDirectory.child
        swapped: list[str] = []

        def swap_then_refuse(self, name):  # type: ignore[no-untyped-def]
            if name == "swapme" and not swapped:
                swapped.append(name)
                target = Path(self.path) / name
                for leftover in target.iterdir():
                    leftover.unlink()
                target.rmdir()
                make_dir_link(target, victim)
                raise OSError(refusal_errno, "refused by the pinned open")
            return real_child(self, name)

        with mock.patch.object(platform_compat.PinnedDirectory, "child", swap_then_refuse):
            artifacts.ArtifactStore._rmtree(doomed)

        assert swapped, "the fixture never swapped the child, so nothing was proved"
        assert not doomed.exists(), "the link was left behind, so the root could not go"
        assert (victim / "precious.txt").read_text(encoding="utf-8") == (
            "not the caller's to delete"
        )
        assert victim.is_dir()

    def test_a_directory_that_cannot_be_opened_is_not_mistaken_for_a_link(
        self, tmp_path: Path
    ) -> None:
        """The other half of dispatching on the name: a REAL directory that refuses
        to open is a failure, not a link to unlink."""
        doomed = tmp_path / "doomed"
        keep = doomed / "keep"
        keep.mkdir(parents=True)
        (keep / "own.txt").write_text("mine", encoding="utf-8")

        real_child = platform_compat.PinnedDirectory.child

        def refuse_that_one(self, name):  # type: ignore[no-untyped-def]
            if name == "keep":
                raise PermissionError(errno.EACCES, "refused")
            return real_child(self, name)

        with mock.patch.object(platform_compat.PinnedDirectory, "child", refuse_that_one):
            with pytest.raises(OSError):
                artifacts.ArtifactStore._rmtree(doomed)

        assert (keep / "own.txt").exists(), "a directory that only refused to open was emptied"


class TestSnapshotImportDoesNotStripThroughALink:
    """``portability._strip_host_local_store_state`` prunes host-local state.

    ``rglob`` walked through a junction planted in a hand-built archive, so an
    entry OUT THERE whose relative name matched the predicate was removed; the
    same link also reached ``shutil.rmtree``, which refuses a reparse point and
    raised out of the import.
    """

    def test_a_link_is_not_walked_and_the_import_does_not_raise(self, tmp_path: Path) -> None:
        victim = tmp_path / "victim"
        # ``<store>/backups`` is host-local state, so walking through the link and
        # judging what is behind it by relative name would delete this.
        (victim / "backups").mkdir(parents=True)
        (victim / "backups" / "precious.db").write_text("outside", encoding="utf-8")

        snap = tmp_path / "snap"
        stores = snap / portability.MEMORY_STORES_DIR_NAME
        (stores / "real-store" / "backups").mkdir(parents=True)
        (stores / "real-store" / "backups" / "old.db").write_text("inside", encoding="utf-8")
        (stores / "real-store" / "memory.md").write_text("# mine", encoding="utf-8")
        make_dir_link(stores / "linked-store", victim)

        portability._strip_host_local_store_state(snap)

        assert not (
            stores / "real-store" / "backups"
        ).exists(), "host-local state inside the archive is still stripped"
        assert (stores / "real-store" / "memory.md").exists(), "the memory itself was removed"
        assert (victim / "backups" / "precious.db").read_text(
            encoding="utf-8"
        ) == "outside", "state behind the link was deleted"

    def test_a_host_local_root_entry_is_still_removed(self, tmp_path: Path) -> None:
        """The other arm of the predicate, so the rewritten walk keeps both."""
        snap = tmp_path / "snap"
        stores = snap / portability.MEMORY_STORES_DIR_NAME
        (stores / ".execution-logs").mkdir(parents=True)
        (stores / ".execution-logs" / "run.jsonl").write_text("host-local", encoding="utf-8")
        (stores / "real-store").mkdir()
        (stores / "real-store" / "memory.md").write_text("# mine", encoding="utf-8")

        portability._strip_host_local_store_state(snap)

        assert not (stores / ".execution-logs").exists()
        assert (stores / "real-store" / "memory.md").exists()

    def test_a_link_swapped_in_after_the_screen_is_not_descended(self, tmp_path: Path) -> None:
        """The race the pin closes, on the import side.

        The extraction directory is only owner-restricted, which does not exclude a
        same-UID agent process, so the link needs no help from the archive: a
        concurrent writer replaces a store that screened clean before the walk
        reaches it. ``os.walk`` re-checked descent with ``os.path.islink``, False for
        a junction, so it descended and deleted the target's ``backups``.

        The swap is driven from the predicate call, which BOTH the old walk and the
        pinned recursion make for every entry before deciding what to do with it --
        so this test is red against the walk for the right reason (the victim is
        stripped) rather than merely because a patched seam went unused.
        """
        victim = tmp_path / "victim"
        (victim / "backups").mkdir(parents=True)
        (victim / "backups" / "precious.db").write_text("outside", encoding="utf-8")

        snap = tmp_path / "snap"
        stores = snap / portability.MEMORY_STORES_DIR_NAME
        swapme = stores / "swapme"
        (swapme / "backups").mkdir(parents=True)
        (swapme / "backups" / "old.db").write_text("inside", encoding="utf-8")

        real_pred = portability.is_host_local_store_state
        swapped: list[tuple[str, ...]] = []

        def swap_then_judge(rel):  # type: ignore[no-untyped-def]
            if rel == (portability.MEMORY_STORES_DIR_NAME, "swapme") and not swapped:
                swapped.append(rel)
                shutil.rmtree(swapme)
                make_dir_link(swapme, victim)
            return real_pred(rel)

        with mock.patch.object(portability, "is_host_local_store_state", swap_then_judge):
            portability._strip_host_local_store_state(snap)

        assert swapped, "the fixture never swapped the store, so nothing was proved"
        assert (victim / "backups" / "precious.db").read_text(
            encoding="utf-8"
        ) == "outside", "the swapped-in link was descended and its target was stripped"
        assert victim.is_dir(), "the link target itself was removed"
        # Not this function's to delete: the predicate never matched the store name,
        # so a link that arrived at it is left alone rather than removed.
        assert platform_compat.is_link_or_junction(swapme), "the link itself was removed"

    def test_a_host_local_link_swapped_in_is_removed_as_a_link(self, tmp_path: Path) -> None:
        """The other arm: when the predicate DOES match, the link goes -- never its target.

        Against the old walk this raised instead: ``shutil.rmtree`` refuses a reparse
        point, so the swap took the import down rather than stripping the entry.
        """
        victim = tmp_path / "victim"
        victim.mkdir()
        (victim / "precious.db").write_text("outside", encoding="utf-8")

        snap = tmp_path / "snap"
        stores = snap / portability.MEMORY_STORES_DIR_NAME
        store = stores / "real-store"
        (store / "backups").mkdir(parents=True)
        (store / "backups" / "old.db").write_text("inside", encoding="utf-8")

        real_pred = portability.is_host_local_store_state
        swapped: list[tuple[str, ...]] = []
        backups = store / "backups"

        def swap_then_judge(rel):  # type: ignore[no-untyped-def]
            if rel[-2:] == ("real-store", "backups") and not swapped:
                swapped.append(rel)
                shutil.rmtree(backups)
                make_dir_link(backups, victim)
            return real_pred(rel)

        with mock.patch.object(portability, "is_host_local_store_state", swap_then_judge):
            portability._strip_host_local_store_state(snap)

        assert swapped, "the fixture never swapped the directory, so nothing was proved"
        assert not backups.exists(), "the matching link was left behind"
        assert (victim / "precious.db").read_text(
            encoding="utf-8"
        ) == "outside", "the removal went through the link into its target"


class TestTheNodeModulesRemovalReportsTheTruth:
    """``node_modules_txn._gone`` must not wedge on a link it cannot rmtree.

    ``shutil.rmtree`` refuses a reparse point ("Cannot call rmtree on a symbolic
    link") and ``ignore_errors=True`` swallows the refusal, so a junction was
    left in place and ``_gone`` returned False -- the permanent, hand-escapable
    wedge the function's own docstring exists to prevent, reached on Windows by
    the link type that needs no privilege.
    """

    def test_a_linked_tree_is_cleared_and_confirmed(self, tmp_path: Path) -> None:
        victim = _seed_victim(tmp_path)
        link = tmp_path / "node_modules"
        make_dir_link(link, victim)

        assert node_modules_txn._gone(link) is True, "the removal was not confirmed"
        assert not os.path.lexists(link), "the link is still at the path"
        assert (victim / "precious.txt").exists(), "the target was removed with the link"


class TestTheSkillCandidateFenceSeesALink:
    """``_candidate_has_symlink`` refuses a candidate that contains a link.

    With ``os.path.islink`` a junction answered False AND ``os.walk`` descended
    it, so the read/approve paths reached whatever it pointed at.
    """

    @pytest.mark.parametrize("where", ["top", "nested"])
    def test_a_linked_directory_in_the_candidate_is_refused(
        self, tmp_path: Path, where: str
    ) -> None:
        victim = _seed_victim(tmp_path)
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        (candidate / "SKILL.md").write_text("# s", encoding="utf-8")
        if where == "top":
            make_dir_link(candidate / "linked", victim)
        else:
            (candidate / "scripts").mkdir()
            make_dir_link(candidate / "scripts" / "linked", victim)

        assert skills.SkillsLoader._candidate_has_symlink(candidate) is True

    def test_a_clean_candidate_is_still_allowed(self, tmp_path: Path) -> None:
        candidate = tmp_path / "candidate"
        (candidate / "scripts").mkdir(parents=True)
        (candidate / "SKILL.md").write_text("# s", encoding="utf-8")
        (candidate / "scripts" / "run.py").write_text("print(1)\n", encoding="utf-8")

        assert skills.SkillsLoader._candidate_has_symlink(candidate) is False


class TestThePendingDetailReadIsPinned:
    """``get_pending_skill`` reads through pins, so no name it judged is re-resolved.

    The screen this replaces refused a link that was PRESENT at check time, so a
    candidate tree the LLM can write needed only to have the link absent at the
    screen and present by the read a few statements later -- and then the contents
    of a file of its choosing were served, because the redaction passes cover
    credential and exfil-URL shapes only. Every open here refuses a link AT THE
    NAME, so the refusal cannot be outrun.
    """

    @staticmethod
    def _stage(tmp_path: Path):  # type: ignore[no-untyped-def]
        loader = skills.SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False)
        loader.stage_skill_candidate(
            "cand",
            description="desc",
            triggers="cand",
            procedure_md="## Steps\n\nrun it",
            provenance=skills.AutoSkillProvenance(
                session_key="s", created_at="2026-01-01T00:00:00"
            ),
            scripts=[{"filename": "run.py", "content": "print(1)\n"}],
        )
        return loader, loader._pending_root() / "cand"

    def test_a_clean_candidate_still_reads_with_translated_newlines(self, tmp_path: Path) -> None:
        """Parity guard: the pinned read must normalise newlines as ``read_text`` did.

        Reading bytes and decoding them does NOT translate ``\\r\\n``, so a candidate
        staged on Windows would start serving carriage returns through the API.
        """
        loader, _pdir = self._stage(tmp_path)

        detail = loader.get_pending_skill("cand")

        assert detail is not None
        assert "run it" in detail["content"]
        assert "\r" not in detail["content"], "the API is serving carriage returns"
        assert detail["scripts"] == [{"filename": "run.py", "content": "print(1)\n"}]

    def test_a_linked_scripts_directory_is_refused_and_not_read(self, tmp_path: Path) -> None:
        victim = _seed_victim(tmp_path)
        loader, pdir = self._stage(tmp_path)
        shutil.rmtree(pdir / "scripts")
        make_dir_link(pdir / "scripts", victim)

        detail = loader.get_pending_skill("cand")

        assert detail is None, "a candidate whose scripts dir is a link was still served"
        assert (victim / "precious.txt").exists(), "the link target was disturbed"

    def test_a_link_planted_just_before_the_read_cannot_redirect_it(self, tmp_path: Path) -> None:
        """The race the pin closes: the refusal is the OPEN, not a check before it.

        The swap is driven from inside the first pinned read, i.e. after any screen
        would have run and immediately before the entry is reached -- the narrowest
        window an adversary could hope for. It still cannot redirect the read.
        """
        victim = _seed_victim(tmp_path)
        loader, pdir = self._stage(tmp_path)

        real_read_text = platform_compat.PinnedDirectory.read_text
        swapped: list[str] = []

        def swap_then_read(self, name, encoding="utf-8"):  # type: ignore[no-untyped-def]
            if name == "SKILL.md" and not swapped:
                swapped.append(name)
                shutil.rmtree(pdir / "scripts")
                make_dir_link(pdir / "scripts", victim)
            return real_read_text(self, name, encoding)

        with mock.patch.object(platform_compat.PinnedDirectory, "read_text", swap_then_read):
            detail = loader.get_pending_skill("cand")

        assert swapped, "the fixture never swapped the directory, so nothing was proved"
        assert detail is None, "the swapped-in link was read through"
        assert (victim / "precious.txt").read_text(encoding="utf-8") == (
            "not the caller's to delete"
        ), "the link target was modified"

    def test_the_detail_read_never_reaches_for_a_by_path_reader(self, tmp_path: Path) -> None:
        """Structural pin: nothing on this path may re-resolve a name by path.

        The exploit needs a FILE symlink swapped into the window, which an
        unprivileged Windows user cannot create, so no behavioural test on this host
        can catch a future edit that reaches for ``_collect_scripts`` or
        ``_read_pending_meta`` here and reopens the window. Measured against the
        screen-then-read version this replaces: a junction swapped in at
        ``_read_pending_meta`` time made the API serve the bytes of a file outside the
        candidate. So the pin is that those readers are NOT called.
        """
        loader, _pdir = self._stage(tmp_path)

        def refuse(*_a: object, **_k: object) -> None:
            raise AssertionError("the detail read re-resolved a name by path")

        with mock.patch.object(skills.SkillsLoader, "_collect_scripts", staticmethod(refuse)):
            with mock.patch.object(skills.SkillsLoader, "_read_pending_meta", refuse):
                detail = loader.get_pending_skill("cand")

        assert detail is not None, "the candidate stopped being readable at all"
        assert detail["scripts"] == [{"filename": "run.py", "content": "print(1)\n"}]

    def test_a_non_utf8_script_refuses_the_candidate_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        """Bytes the writer of the tree chose must not become a 500.

        ``read_text`` raises ``UnicodeDecodeError``, a ValueError that no caller on
        this path catches, so letting it escape turns a candidate into a failed
        request rather than a refused one.
        """
        loader, pdir = self._stage(tmp_path)
        (pdir / "scripts" / "binary.py").write_bytes(b"\xff\xfe not utf-8 \x00")

        detail = loader.get_pending_skill("cand")

        assert detail is None, "the undecodable candidate was served anyway"

    def test_a_hardlinked_entry_is_refused_by_the_pinned_read(self, tmp_path: Path) -> None:
        """A hardlink shares its target's inode while carrying its own name, so every
        path-based guard is blind to it. The refusal has to come from the descriptor."""
        secret = tmp_path / "secret.txt"
        secret.write_text("CREDENTIAL-BYTES", encoding="utf-8")

        loader, pdir = self._stage(tmp_path)
        alias = pdir / "scripts" / "alias.py"
        try:
            os.link(secret, alias)
        except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform
            pytest.skip(f"hardlinks unavailable here: {exc}")

        assert alias.stat().st_nlink > 1, "the fixture did not create a hardlink"

        detail = loader.get_pending_skill("cand")

        assert detail is None, "a hardlinked entry was read out through the candidate"


class TestTheDeployStagingFenceSeesALink:
    """``deploy.handlers._stage_tree_safe`` refuses a linked directory in the tree.

    The fence read ``Path.is_symlink()``, so a junction passed it. That is NOT a
    way into the deployment: ``safe_read_file_bytes_nolink(within_root=...)``
    pins containment to the opened descriptor, so a file behind the link is
    refused there. What the fence being blind cost is the two things below --
    a link with nothing readable behind it passed the entire walk and was staged
    as an ordinary empty directory, and a link with files behind it was reported
    as ``staging-read-blocked`` rather than as the link it is.
    """

    def test_a_link_with_an_empty_target_no_longer_stages_as_a_plain_dir(
        self, tmp_path: Path
    ) -> None:
        """The escape that WAS reachable: no file behind the link, so the
        downstream containment gate never runs and the walk completed."""
        from kiro_crew.deploy import handlers

        outside = tmp_path / "outside"
        outside.mkdir()
        source = tmp_path / "site"
        source.mkdir()
        (source / "index.html").write_text("<html></html>", encoding="utf-8")
        make_dir_link(source / "assets", outside)

        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(RuntimeError, match="symlink-in-tree"):
            handlers._stage_tree_safe(source, staging)

    def test_a_link_with_files_behind_it_is_named_as_a_link(self, tmp_path: Path) -> None:
        """Blocked either way, but the rejection now names the real cause -- the
        tag chooses the structured response the caller returns."""
        from kiro_crew.deploy import handlers

        victim = _seed_victim(tmp_path)
        source = tmp_path / "site"
        source.mkdir()
        (source / "index.html").write_text("<html></html>", encoding="utf-8")
        make_dir_link(source / "assets", victim)

        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(RuntimeError, match="symlink-in-tree"):
            handlers._stage_tree_safe(source, staging)

        assert not (
            staging / "tree" / "assets"
        ).exists(), "content behind the link reached the staged snapshot"

    def test_a_clean_tree_still_stages(self, tmp_path: Path) -> None:
        from kiro_crew.deploy import handlers

        source = tmp_path / "site"
        (source / "assets").mkdir(parents=True)
        (source / "index.html").write_text("<html></html>", encoding="utf-8")
        (source / "assets" / "app.css").write_text("body{}", encoding="utf-8")

        staging = tmp_path / "staging"
        staging.mkdir()
        staged = handlers._stage_tree_safe(source, staging)

        assert (staged / "index.html").read_text(encoding="utf-8") == "<html></html>"
        assert (staged / "assets" / "app.css").read_text(encoding="utf-8") == "body{}"
