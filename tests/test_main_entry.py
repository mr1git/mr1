from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

import main as main_entry


def test_main_help_exposes_tui_and_not_legacy_flags(capsys):
    with patch.object(sys, "argv", ["main.py", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main_entry.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--tui" in out
    assert "--plain" in out
    assert "--web" not in out
    assert "--termui" not in out


def test_main_tui_dispatches_to_new_entrypoint():
    with patch("mr1.tui.main", return_value=0) as mock_tui_main:
        with patch.object(sys, "argv", ["main.py", "--tui"]):
            with pytest.raises(SystemExit) as exc:
                main_entry.main()
    assert exc.value.code == 0
    mock_tui_main.assert_called_once_with()
