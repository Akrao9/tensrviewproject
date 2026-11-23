"""SIGTERM sent the moment the server says it is listening must stop it cleanly (the author, Nov 23, 2026).

Found on Linux CI (981fe6c, `test_lane_ab2_server`: a restarted server ended with -15): the server printed its
listening line, and started the task that writes each run's status, before it asked for SIGTERM. A SIGTERM
in that window met the default action and killed the process with nothing written and no stop recorded; one
that arrived while the handler was being registered could be dropped, and the server went on running.
"""

from __future__ import annotations

import os
import signal
import subprocess

import pytest
from test_flight_e2e import BINARY

pytestmark = pytest.mark.skipif(BINARY is None or not hasattr(signal, "SIGTERM") or os.name != "posix",
                                reason="needs the built tensorquill-server and POSIX signals")

ROUNDS = 60  # the window is a few instructions wide: before the fix about 6 starts in 100 died here (macOS, release)


def test_sigterm_right_after_the_listening_line_stops_the_server_cleanly(tmp_path):
    """The real process, no sleep anywhere: read the listening line, send SIGTERM at once, and the server
    must take its ordinary stop (exit status 0), every time. The wait is a bound, not a timing asserted."""
    env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
    ended = []
    for round_ in range(ROUNDS):
        store = tmp_path / f"store-{round_}"
        process = subprocess.Popen([str(BINARY), "--listen", "127.0.0.1:0", "--store", str(store)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        try:
            line = process.stdout.readline()
            process.send_signal(signal.SIGTERM)
            assert line.startswith("tensorquill-server listening on "), (line, process.stderr.read())
            try:
                ended.append(process.wait(30))
            except subprocess.TimeoutExpired:
                ended.append("still running 30 s after SIGTERM")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(30)
            process.stdout.close()
            process.stderr.close()
    assert ended == [0] * ROUNDS, {str(code): ended.count(code) for code in set(ended)}
