import json
import os

import pytest

from tradelab.state import StateStore, StateUnreadableError


def test_first_start_has_no_state(tmp_path):
    assert StateStore(tmp_path / "s.json").load() is None


def test_roundtrip(tmp_path):
    s = StateStore(tmp_path / "s.json")
    s.save({"trades": 3, "pnl": 1.5})
    assert s.load() == {"trades": 3, "pnl": 1.5}


def test_previous_state_is_kept_as_backup(tmp_path):
    s = StateStore(tmp_path / "s.json")
    s.save({"v": 1})
    s.save({"v": 2})
    assert json.loads(s.backup.read_text()) == {"v": 1} and s.load() == {"v": 2}


def test_no_temporary_file_is_left_behind(tmp_path):
    s = StateStore(tmp_path / "s.json")
    s.save({"v": 1})
    s.save({"v": 2})
    assert not s.tmp.exists()


def test_corrupted_main_file_falls_back_to_the_backup(tmp_path):
    s = StateStore(tmp_path / "s.json")
    s.save({"v": 1})
    s.save({"v": 2})
    s.path.write_text("{ ceci n'est pas du json", encoding="utf-8")
    assert s.load() == {"v": 1}


def test_unreadable_state_refuses_to_start_empty(tmp_path):
    """Repartir à vide alors que le compte contient des positions fait racheter en boucle."""
    s = StateStore(tmp_path / "s.json")
    s.save({"v": 1})
    s.save({"v": 2})
    s.path.write_text("???", encoding="utf-8")
    s.backup.write_text("", encoding="utf-8")
    with pytest.raises(StateUnreadableError):
        s.load()


def test_a_json_value_that_is_not_an_object_counts_as_unreadable(tmp_path):
    s = StateStore(tmp_path / "s.json")
    s.path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(StateUnreadableError):
        s.load()


def test_a_crash_during_the_write_leaves_the_old_state_intact(tmp_path, monkeypatch):
    s = StateStore(tmp_path / "s.json")
    s.save({"v": 1})

    def boom(fd):
        raise OSError("coupure pendant l'écriture")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        s.save({"v": 2})
    monkeypatch.undo()
    assert s.load() == {"v": 1}


def test_non_serialisable_values_are_stringified(tmp_path):
    import datetime

    s = StateStore(tmp_path / "s.json")
    s.save({"when": datetime.datetime(2026, 1, 2, 3, 4)})
    assert s.load()["when"].startswith("2026-01-02")
