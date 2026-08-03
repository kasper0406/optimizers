"""git_provenance dirty semantics: untracked data products are not code drift.

Local sequential sweeps append untracked results/*.json between runs, and
data/ is a download cache; neither may flag later runs of the same sweep
git-dirty. Modified tracked files — anywhere, including results/ — and
untracked files outside those two directories still do.
"""

import subprocess

import pytest

from src.results_io import git_provenance


@pytest.fixture()
def repo(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            env={
                "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                "PATH": "/usr/bin:/bin",
                "HOME": str(tmp_path),
            },
        )

    git("init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "old.json").write_text("{}\n")
    git("add", "code.py", "results/old.json")
    git("commit", "-q", "-m", "init")
    return tmp_path


def test_clean_tree(repo):
    prov = git_provenance(repo)
    assert prov["git_dirty"] is False
    assert len(prov["git_sha"]) == 40


def test_untracked_results_and_data_are_not_dirty(repo):
    (repo / "results" / "new.json").write_text("{}\n")
    (repo / "data").mkdir()
    (repo / "data" / "blob.bin").write_bytes(b"\x00")
    assert git_provenance(repo)["git_dirty"] is False


def test_untracked_code_is_dirty(repo):
    (repo / "new_module.py").write_text("y = 2\n")
    assert git_provenance(repo)["git_dirty"] is True


def test_modified_tracked_code_is_dirty(repo):
    (repo / "code.py").write_text("x = 2\n")
    assert git_provenance(repo)["git_dirty"] is True


def test_modified_tracked_results_file_is_dirty(repo):
    # append-only violation must still surface
    (repo / "results" / "old.json").write_text('{"edited": true}\n')
    assert git_provenance(repo)["git_dirty"] is True
