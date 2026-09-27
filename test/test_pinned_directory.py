"""``platform_compat.PinnedDirectory``: act on the directory you inspected.

The class exists because the two platforms reach that property by OPPOSITE routes,
and a caller written in terms of one is broken on the other:

* POSIX has ``dir_fd``-relative operations and the pin does NOT block a rename, so
  the descriptor must be used for everything.
* Windows has no ``dir_fd`` operations at all, so everything goes by path -- sound
  only because the pin makes the path stable (no ``FILE_SHARE_DELETE``, so the
  directory and every ancestor refuse a rename and a delete while it lives).

So these tests assert the BEHAVIOUR both routes are for, plus one test per
platform for the mechanism that route depends on. Links are staged through
``conftest.make_dir_link`` -- a symlink on POSIX, a real junction on Windows.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from conftest import make_dir_link
from kiro_crew import platform_compat


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "sub" / "b.txt").write_text("b", encoding="utf-8")
    return root


class TestWhatItReadsAndRemoves:
    def test_it_lists_and_classifies_its_own_entries(self, tree: Path) -> None:
        with platform_compat.pinned_directory(tree) as pinned:
            assert sorted(pinned.names()) == ["a.txt", "sub"]
            assert pinned.is_dir("sub") is True
            assert pinned.is_dir("a.txt") is False
            assert pinned.is_link("sub") is False
            assert stat.S_ISREG((pinned.lstat("a.txt") or os.stat_result).st_mode)

    def test_an_unreadable_name_lstats_as_none_rather_than_raising(self, tree: Path) -> None:
        with platform_compat.pinned_directory(tree) as pinned:
            assert pinned.lstat("not-there") is None
            assert pinned.is_dir("not-there") is False
            assert pinned.is_link("not-there") is False

    def test_it_removes_a_file_and_an_empty_directory(self, tree: Path) -> None:
        with platform_compat.pinned_directory(tree) as pinned:
            with pinned.child("sub") as sub:
                sub.unlink("b.txt")
            pinned.rmdir("sub")
            pinned.unlink("a.txt")
            assert pinned.names() == []
        assert tree.is_dir() and not any(tree.iterdir())


class TestALinkIsSeenAndNeverTraversed:
    def test_a_directory_link_is_classified_as_a_link_not_a_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        (target / "precious.txt").write_text("outside", encoding="utf-8")
        holder = tmp_path / "holder"
        holder.mkdir()
        make_dir_link(holder / "link", target)

        with platform_compat.pinned_directory(holder) as pinned:
            assert pinned.is_link("link") is True
            assert pinned.is_dir("link") is False, "a link must not answer as its target's shape"

    def test_child_refuses_a_link_in_the_open_itself(self, tmp_path: Path) -> None:
        """The refusal is the open, not a check before it -- that is the point."""
        target = tmp_path / "target"
        target.mkdir()
        holder = tmp_path / "holder"
        holder.mkdir()
        make_dir_link(holder / "link", target)

        with platform_compat.pinned_directory(holder) as pinned:
            with pytest.raises(OSError):
                pinned.child("link").close()

    def test_pinned_directory_refuses_a_link_at_the_top(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        make_dir_link(link, target)

        with pytest.raises(OSError):
            platform_compat.pinned_directory(link).close()

    def test_removing_a_link_leaves_its_target_alone(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        (target / "precious.txt").write_text("outside", encoding="utf-8")
        holder = tmp_path / "holder"
        holder.mkdir()
        make_dir_link(holder / "link", target)

        with platform_compat.pinned_directory(holder) as pinned:
            pinned.unlink("link")
            assert pinned.names() == []
        assert (target / "precious.txt").read_text(encoding="utf-8") == "outside"
        assert target.is_dir()


class TestTheMechanismEachPlatformDependsOn:
    """One test per platform for the property that route rests on.

    Guards against the class silently degrading to by-name operations with no
    protection at all: on POSIX that would mean the descriptor is not being used,
    and on Windows that the share-mode lock is not being taken.
    """

    @pytest.mark.skipif(platform_compat.IS_WINDOWS, reason="dir_fd operations are POSIX-only")
    def test_on_posix_the_descriptor_survives_a_rename_of_the_name(self, tree: Path) -> None:
        """POSIX does not block the rename, so the descriptor is what must carry."""
        with platform_compat.pinned_directory(tree) as pinned:
            moved = tree.parent / "moved"
            os.rename(tree, moved)
            try:
                # The name is gone, so anything by-name would fail or hit another
                # object. Reading through the pin still describes the same directory.
                assert sorted(pinned.names()) == ["a.txt", "sub"]
                pinned.unlink("a.txt")
                assert sorted(pinned.names()) == ["sub"]
            finally:
                os.rename(moved, tree)

    @pytest.mark.skipif(not platform_compat.IS_WINDOWS, reason="share-mode locking is Windows-only")
    def test_on_windows_the_pin_blocks_renaming_the_directory_and_its_parent(
        self, tree: Path
    ) -> None:
        """Windows has no dir_fd, so by-path is sound only while this holds."""
        with platform_compat.pinned_directory(tree) as pinned:
            assert pinned.names(), "fixture is empty, so the pin proves nothing"
            with pytest.raises(OSError):
                os.rename(tree, tree.parent / "moved")
            with pytest.raises(OSError):
                os.rename(tree.parent, tree.parent.parent / "parent-moved")

    def test_a_parent_stays_usable_while_a_child_is_pinned(self, tree: Path) -> None:
        """A chain of pins is what anchors a whole path, so both must be live."""
        with platform_compat.pinned_directory(tree) as parent:
            with parent.child("sub") as child:
                assert child.names() == ["b.txt"]
                assert sorted(parent.names()) == ["a.txt", "sub"]

    def test_the_descriptor_is_closed_on_exit(self, tree: Path) -> None:
        pinned = platform_compat.pinned_directory(tree)
        with pinned:
            fd = pinned.fd
            assert os.fstat(fd).st_mode
        with pytest.raises(OSError):
            os.fstat(fd)
