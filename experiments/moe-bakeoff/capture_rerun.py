"""Capture the stock Rerun viewer on an MoE fixture at a known training step."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from rerun.experimental import ViewerClient


VIEWS = {"Training incidents", "Routing status by layer and expert", "Router diversity by layer"}


def loaded(state, last_step):
    """The blueprint's views exist and the whole recording, up to its last step, has arrived."""
    timelines = [timeline for recording in state.recordings for timeline in recording.timelines
                 if timeline.name == "training_step"]
    names = {view.name for view in state.views}
    return (VIEWS <= names and any(name.endswith(" expert load") for name in names)
            and not state.loading and any(timeline.end == last_step for timeline in timelines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", default="rerun")
    parser.add_argument("--port", type=int, default=9887)
    parser.add_argument("--fixture", default="seeded", help="an MoE fixture, by name")
    parser.add_argument("--step", type=int, help="default: the first incident's step")
    parser.add_argument("--last-step", type=int, help="the recording's final step (default: the last sample)")
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / "artifacts" / args.fixture
    evidence = json.loads((output / f"{args.fixture}.evidence.json").read_text(encoding="utf-8"))
    if not evidence["samples"]:
        parser.error(f"{args.fixture} has no MoE samples, and this capture shows the MoE views")
    last_step = args.last_step or max(sample["step"] for sample in evidence["samples"])
    step = args.step or next(i for i in evidence["incidents"] if i["kind"] == "moe_imbalance")["step"]
    with subprocess.Popen([
        args.executable, "--headless", "--integration-test", "--bind", "127.0.0.1",
        "--port", str(args.port), "--window-size", "1160x720", str(output / f"{args.fixture}.rrd"),
    ], start_new_session=True) as process:
        viewer = ViewerClient.connect(f"rerun+http://127.0.0.1:{args.port}/proxy")
        deadline = time.monotonic() + 20
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Rerun exited with code {process.returncode}")
                try:
                    state = viewer.get_state()
                except RuntimeError:
                    time.sleep(0.1)
                    continue
                if loaded(state, last_step):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("Rerun did not finish loading the MoE recording and blueprint")
            viewer.set_time("training_step", sequence=step, play_state="paused")
            # A screenshot taken straight after the views appear can show time-varying views as
            # "(empty)", so wait for the time cursor and then for two identical frames in a row.
            while viewer.get_state().recordings[0].current_time != step:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Rerun did not move to step {step}")
                time.sleep(0.1)
            previous = None
            while (png := viewer.take_screenshot(max_size=1160)) != previous:
                if time.monotonic() > deadline:
                    raise RuntimeError("Rerun's screenshot did not settle")
                previous = png
                time.sleep(0.5)
            (output / "rerun.png").write_bytes(png)
            print(f"wrote {output / 'rerun.png'} at step {step}")
        finally:
            # The pip-installed `rerun` is a Python wrapper that starts the viewer as a child, so
            # stop the whole process group; otherwise the viewer keeps the port, and the next run
            # streams into the stale viewer and exits.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=3)
                    break
                except subprocess.TimeoutExpired:
                    continue


if __name__ == "__main__":
    main()
