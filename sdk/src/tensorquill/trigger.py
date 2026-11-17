"""The trigger of `--watch`: start a command, post to a webhook, or both, once for each new incident.

    python -m tensorquill run.jsonl --watch --compact \\
        --on-incident 'my-agent "Explain the $TENSORQUILL_INCIDENT_KIND incident in $TENSORQUILL_EVIDENCE"'
    python -m tensorquill store/runs/r --rank all --watch --webhook https://example.com/hooks/tensorquill

**What fires.** An incident is new when its rank, kind and step have not fired before. By
default the invariants (`evidence.INCIDENT_CLASSES`) fire, and so does a stall that has reached
`critical`: a `job_stalled` or `rank_silent` incident (`STALL_KINDS`) whose `severity` is
`critical`, whatever its `assessment`. `--fire-anomalies` adds the anomalies, whose thresholds
are not tuned on other people's jobs yet, and `--fire-kind KIND` adds one kind by name, which is
the only way any other observation, a stall still at `warning`, or a kind this version does not
know fires. A change of run state never fires anything: incidents never close, so the state says
nothing an incident did not say first. In the all-ranks view an incident that several ranks share
(one incident group) fires once for the ranks that show it, not once a rank; ranks that join the
group on a later look fire once more, for those ranks.

**A stall** is the server's reading of what a run's ranks send (`watchdog.json`), so only a
watcher of a server's store, or of a file given a status with `watchdog=`, ever lists one. The
two kinds stay observations: the run state does not move for them, and the class the command
gets is `observation`. The server calls a stopped rank `critical` after its `--critical-seconds`
(900) or three of the rank's own silence thresholds, if that is longer, and gives a severity only
to a rank that is silent, unresponsive or lost: a rank in a declared evaluation or checkpoint
phase is `paused`, has none, and so cannot make a stall `critical`. A stall named with
`--fire-kind` fires at `warning` and not again when it becomes `critical`, since the key is the
same; one not named fires once, when it becomes `critical`. A stall whose kind or step changes
is another incident and fires again: a silent rank at `critical` whose job then stops altogether
(`rank_silent`, then `job_stalled`), or a second rank that falls silent at an earlier step.

**What has fired** is kept in a small state file beside the evidence (`run.evidence.json` and
`run.evidence.compact.json` both give `run.evidence.fired.json`), so a watcher that is
restarted does not fire again. Delete the file to start over.

**The command** is one string, run by the shell. It gets:

- on standard input, one JSON document, the same one the webhook posts (below);
- in its environment, `TENSORQUILL_EVIDENCE` (the evidence file's absolute path),
  `TENSORQUILL_RUN_ID`, `TENSORQUILL_INCIDENT_ID`, `TENSORQUILL_INCIDENT_KIND`,
  `TENSORQUILL_INCIDENT_CLASS`, `TENSORQUILL_INCIDENT_STEP`, `TENSORQUILL_INCIDENT_RANKS`
  (`0`, `0-3, 7`, or empty for an incident of the whole run) and `TENSORQUILL_INCIDENT_SUMMARY`.

Nothing is added to the command line: parameter and module names come from the job, and text
from a job has no place in a string a shell parses. (Child processes get their context through
the environment in `pytorch-lightning`'s launcher too; a document on standard input is how
hooks that carry structured data are fed.) The command is started and not waited for, so an
agent that takes ten minutes does not hold up the watcher, which goes on keeping the evidence
file current under it. It runs in a session of its own: Ctrl-C on the watcher leaves it
running, and nothing here ever kills it. One command runs at a time and the others wait their
turn (after `COMMAND_WAIT_SECONDS` the next is started beside one that is still running). The
incident counts as fired once its command has started, which is what a restarted watcher
goes by; a command that cannot be started or exits with a status other than 0 is a failed try.

**The webhook** is one `POST` of `application/json` with the standard library, `--webhook-timeout`
seconds for the whole exchange (10 by default), to http or https only. A redirect is not
followed, since it leads to a host the user did not name, and counts as a failure; any 2xx
answer is a delivery. Headers: `X-TensorQuill-Event: incident` and `X-TensorQuill-Delivery`,
the same id on every try (as `mlflow`'s webhooks carry a delivery id), so a receiver can drop
a repeat. The body:

    {"event": "incident", "version": 1, "id": "<run_id>/rank0/non_finite/412", "attempt": 1,
     "sent_at_ms": ..., "run_id": ..., "evidence": "<absolute path>", "run_state": "unhealthy",
     "kind": "non_finite", "class": "invariant", "step": 412, "rank": 0, "ranks": [0],
     "new_ranks": [0], "summary": "<one line>", "incident": {<as in the evidence file>}}

`rank` is null and `ranks` lists them when several ranks share the incident; `ranks` is empty
for an incident of the whole run. Numbers that are not finite are null, so the body is strict JSON.

**Failures never reach the watcher.** A try that fails is printed and made again on a later
look, `ATTEMPTS` tries in all (the first and three more, as `mlflow` retries a webhook three
times), `RETRY_SECONDS` apart and doubling; then the incident is given up, which is printed
and kept in the state file. After the first failed post of a look the rest wait for the next,
so an unreachable host costs one timeout a look. The command and the webhook are counted
apart: a webhook that fails does not start the command again.

Nothing is sent or started unless the user gave the command or the URL, and nothing here
touches the training job.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .evidence import ALL_RANKS_VIEW, INCIDENT_CLASSES, _group_line, _rank_ranges, incident_class

STATE_VERSION = 1
PAYLOAD_VERSION = 1
FIRE_CLASSES = ("invariant",)  # the classes that fire with no option
# The two stall kinds, read from the server's status. Observations, and still fired with no
# option once their severity is this: a hung job is what the watcher is left running for.
STALL_KINDS = ("job_stalled", "rank_silent")
STALL_FIRES_AT = "critical"
ATTEMPTS = 4  # tries in all for one incident and one target
RETRY_SECONDS = 5.0  # before the second try; doubled before each one after
WEBHOOK_TIMEOUT = 10.0
# How long the next incident's command waits for the one that is running.
COMMAND_WAIT_SECONDS = 600.0
_OPEN = ("pending", "failed")  # a target that still has a try to make


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # urllib then raises the 3xx as an HTTPError


def state_path_for(evidence_path: str | Path) -> Path:
    """The state file beside an evidence file: `run.evidence.json` and
    `run.evidence.compact.json` both give `run.evidence.fired.json`."""
    path = Path(evidence_path)
    stem = path.name[: -len(".json")] if path.name.endswith(".json") else path.name
    stem = stem[: -len(".compact")] if stem.endswith(".compact") else stem
    return path.with_name(stem + ".fired.json")


def _strict(value: Any) -> Any:
    """`value` with every NaN and Inf as None, for a body any JSON parser reads."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(v) for v in value]
    return value


def _step_key(step: Any) -> Any:
    return step if step is None or isinstance(step, (int, str)) else str(step)


def _who(ranks: Iterable[Any]) -> str:
    numbers = [r for r in ranks if isinstance(r, int)]
    if not numbers:
        return "the run"
    return f"rank{'s' if len(numbers) != 1 else ''} {_rank_ranges(numbers, 8)}"


class Trigger:
    """Fires once for each new incident of the evidence a watcher writes (see the module's text).

    `watch(..., trigger=Trigger("my-command"))` calls `tick()` on every look and
    `look(evidence, path)` after every write; neither ever raises. Invariants fire, and a
    stall at `critical` (`STALL_KINDS`); `anomalies` adds the anomalies, `kinds` any kinds by
    name, a stall at `warning` among them. `state_path` replaces the state file beside the
    evidence. `log` gets each line this prints.
    """

    def __init__(
        self,
        command: str | None = None,
        webhook: str | None = None,
        *,
        anomalies: bool = False,
        kinds: Iterable[str] = (),
        state_path: str | Path | None = None,
        attempts: int = ATTEMPTS,
        retry_seconds: float = RETRY_SECONDS,
        webhook_timeout: float = WEBHOOK_TIMEOUT,
        command_wait: float = COMMAND_WAIT_SECONDS,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if not command and not webhook:
            raise ValueError("a trigger needs a command (--on-incident) or a webhook URL (--webhook)")
        if command is not None and (not isinstance(command, str) or not command.strip()):
            raise ValueError("--on-incident takes the command as one string, run by the shell")
        self._host = ""
        if webhook is not None:
            try:
                parts = urllib.parse.urlsplit(webhook)
                host, port = parts.hostname, parts.port
            except ValueError:
                parts, host, port = None, None, None
            if parts is None or parts.scheme not in ("http", "https") or not host:
                # The URL is not repeated: a webhook's URL is often its secret.
                raise ValueError("--webhook takes an http:// or https:// URL that names a host")
            self._host = host + (f":{port}" if port else "")
        if attempts < 1 or webhook_timeout <= 0:
            raise ValueError("attempts is at least 1 and the webhook timeout more than 0 seconds")
        self.command, self.webhook = command, webhook
        self.classes = {*FIRE_CLASSES, *(("anomaly",) if anomalies else ())}
        self.kinds = {str(kind) for kind in kinds}
        self.attempts, self.retry_seconds = attempts, retry_seconds
        self.webhook_timeout, self.command_wait = webhook_timeout, command_wait
        self._log = log or (lambda text: print(text, flush=True))
        self._opener = urllib.request.build_opener(_NoRedirect)
        self._state_path = Path(state_path) if state_path is not None else None
        self._loaded = False
        self._deliveries: list[dict[str, Any]] = []
        self._fired: set[tuple[Any, Any, Any]] = set()
        self._units: list[dict[str, Any]] = []
        self._evidence_path: Path | None = None
        self._run_id = ""
        self._run_state: Any = None
        self._commands: list[dict[str, Any]] = []
        self._retry_at: dict[tuple[str, str], float] = {}
        self._unknown: set[str] = set()
        self._save_failed = False

    # --- What the watcher calls ------------------------------------------------------------

    def look(self, evidence: Mapping[str, Any], path: str | Path, *, rank: int | None = None) -> None:
        """Take the evidence just written to `path`: fire for what is new in it. `rank` is the
        rank a one-rank evidence is of (the all-ranks view names its own). Never raises."""
        try:
            self._evidence_path = Path(path).resolve()
            if not self._loaded:
                if self._state_path is None:
                    self._state_path = state_path_for(self._evidence_path)
                self._load()
            self._run_id = str(evidence.get("run_id") or self._evidence_path.name)
            found = evidence.get("run_state")
            self._run_state = found.get("state") if isinstance(found, Mapping) else None
            if rank is None and isinstance(evidence.get("detection"), Mapping):
                rank = evidence["detection"].get("rank")
            self._units = self._collect(evidence, rank)
            self._enqueue()
            self._work()
        except Exception as error:  # a trigger never stops the watcher
            self._log(f"trigger: {type(error).__name__}: {error}; the watcher goes on")

    def tick(self) -> None:
        """A look of the watcher, written or not: see how started commands ended, and make
        the tries that are due. A run that died at its bad step writes nothing more, so
        retries cannot wait for the next write. Never raises."""
        try:
            self._work()
        except Exception as error:
            self._log(f"trigger: {type(error).__name__}: {error}; the watcher goes on")

    def close(self, wait: float = 0.0, kill: bool = False) -> None:
        """For scripts and tests; the watcher never calls it, so a command outlives the watcher.
        Wait up to `wait` seconds for the commands that are running and record how they
        ended; with `kill`, stop those still running after that."""
        deadline = time.monotonic() + wait
        while True:
            self._reap()
            if not self._commands or time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        if kill:
            for item in self._commands:
                process = item["process"]
                try:
                    os.killpg(process.pid, signal.SIGTERM) if hasattr(os, "killpg") else process.terminate()
                except OSError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            self._commands.clear()
        self._save()

    @property
    def deliveries(self) -> list[dict[str, Any]]:
        """What has fired, as the state file holds it: one entry for each firing."""
        return json.loads(json.dumps(self._deliveries))

    # --- From evidence to firings ----------------------------------------------------------

    def _fires(self, incident: Mapping[str, Any]) -> bool:
        """Whether the options let this incident fire: its kind was named, its class fires, or
        it is a stall the server calls critical. The severity is the incident's own, which
        only a stopped rank sets (`evidence._progress_incidents`, and `progress_incidents` in
        the server's `watchdog.rs`): a paused rank has none. One with no severity does not fire."""
        kind = incident.get("kind")
        if kind in self.kinds or incident_class(kind) in self.classes:
            return True
        return kind in STALL_KINDS and incident.get("severity") == STALL_FIRES_AT

    def _unit(self, incident: Mapping[str, Any], ranks: list[Any], group: Mapping[str, Any]) -> dict[str, Any] | None:
        kind = incident.get("kind")
        if not self._fires(incident):
            if incident_class(kind) is None and isinstance(kind, str) and kind not in self._unknown:
                self._unknown.add(kind)
                self._log(f"trigger: incident kind {kind!r} is not known to this version, so it has no class and"
                          f" does not fire; name it to fire on it (--fire-kind {kind})")
            return None
        try:
            summary = _group_line(group)
        except Exception:  # an incident shaped as this version does not expect
            summary = f"Incident at step {incident.get('step')}: {kind}"
        return {"kind": kind, "class": incident_class(kind), "step": _step_key(incident.get("step")),
                "ranks": ranks, "incident": incident, "summary": summary}

    def _collect(self, evidence: Mapping[str, Any], rank: int | None) -> list[dict[str, Any]]:
        """The incidents of `evidence` that the options let fire, each with the ranks it is on."""
        units = []
        if evidence.get("view") == ALL_RANKS_VIEW:
            for group in evidence.get("incident_groups") or ():
                if isinstance(group, Mapping) and isinstance(group.get("incident"), Mapping):
                    units.append(self._unit(group["incident"], list(group.get("ranks") or ()), group))
            shared = evidence.get("run_incidents") or ()
        else:
            shared = evidence.get("incidents") or ()
        for incident in shared:
            if isinstance(incident, Mapping):
                units.append(self._unit(incident, [rank], {
                    "incident": incident, "kind": incident.get("kind"), "step": incident.get("step"),
                    "ranks": [rank], "incident_rank": rank, "numbers_differ": False,
                }))
        return [unit for unit in units if unit is not None]

    def _enqueue(self) -> None:
        """A firing for each incident with a rank that has not fired before, for those ranks."""
        targets = [name for name, given in (("command", self.command), ("webhook", self.webhook)) if given]
        for unit in self._units:
            new = [r for r in unit["ranks"] if (r, unit["kind"], unit["step"]) not in self._fired]
            if not new:
                continue
            numbers = [r for r in new if isinstance(r, int)]
            who = f"rank{'s' if len(numbers) != 1 else ''}{_rank_ranges(numbers).replace(' ', '')}" if numbers else "run"
            self._deliveries.append({
                "id": f"{self._run_id}/{who}/{unit['kind']}/{unit['step']}",
                "kind": unit["kind"], "step": unit["step"], "ranks": new,
                "first_seen_ms": int(time.time() * 1000),
                **{target: {"status": "pending", "attempts": 0} for target in targets},
            })
            self._fired.update((r, unit["kind"], unit["step"]) for r in new)
        self._save()

    def _unit_for(self, delivery: Mapping[str, Any]) -> dict[str, Any] | None:
        """The incident a firing is for, in the evidence last looked at; None once a resumed
        run has rolled it back, and then nothing is sent for it."""
        ranks = set(delivery["ranks"])
        for unit in self._units:
            if unit["kind"] == delivery["kind"] and unit["step"] == delivery["step"] and ranks & set(unit["ranks"]):
                return unit
        return None

    # --- Making the tries ------------------------------------------------------------------

    def _due(self, delivery: Mapping[str, Any], target: str) -> bool:
        entry = delivery.get(target)
        if not isinstance(entry, dict) or entry.get("status") not in _OPEN:
            return False
        return time.monotonic() >= self._retry_at.get((delivery["id"], target), 0.0)

    def _work(self) -> None:
        self._reap()
        if self.webhook:
            for delivery in self._deliveries:
                if not self._due(delivery, "webhook"):
                    continue
                unit = self._unit_for(delivery)
                if unit is not None and not self._post(delivery, unit):
                    break  # one failed post a look: the rest wait, so a dead host costs one timeout
        if self.command and not any(item["holds"] for item in self._commands):
            for delivery in self._deliveries:
                unit = self._unit_for(delivery) if self._due(delivery, "command") else None
                if unit is not None:
                    self._start(delivery, unit)
                    break  # one command at a time
        self._save()

    def _label(self, delivery: Mapping[str, Any]) -> str:
        return f"trigger: {delivery['kind']} at step {delivery['step']} ({_who(delivery['ranks'])})"

    def _payload(self, delivery: Mapping[str, Any], unit: Mapping[str, Any], target: str) -> dict[str, Any]:
        ranks = [r for r in unit["ranks"] if isinstance(r, int)]
        return _strict({
            "event": "incident",
            "version": PAYLOAD_VERSION,
            "id": delivery["id"],
            "attempt": delivery[target]["attempts"],
            "sent_at_ms": int(time.time() * 1000),
            "run_id": self._run_id,
            "evidence": str(self._evidence_path),
            "run_state": self._run_state,
            "kind": unit["kind"],
            "class": unit["class"],
            "step": unit["step"],
            "rank": ranks[0] if len(ranks) == 1 else None,
            "ranks": ranks,
            "new_ranks": [r for r in delivery["ranks"] if isinstance(r, int)],
            "summary": unit["summary"],
            "incident": unit["incident"],
        })

    def _failed(self, delivery: dict[str, Any], target: str, what: str) -> None:
        entry = delivery[target]
        entry["error"] = what
        tried = entry["attempts"]
        if tried >= self.attempts:
            entry["status"] = "gave_up"
            self._log(f"{self._label(delivery)}: {what} (try {tried} of {self.attempts}); given up. To try again,"
                      f" stop the watcher and remove this incident from {self._state_path}")
        else:
            entry["status"] = "failed"
            wait = self.retry_seconds * 2 ** (tried - 1)
            self._retry_at[(delivery["id"], target)] = time.monotonic() + wait
            self._log(f"{self._label(delivery)}: {what} (try {tried} of {self.attempts}); tried again"
                      f" on a later look, in {wait:g} s or more")

    def _post(self, delivery: dict[str, Any], unit: Mapping[str, Any]) -> bool:
        entry = delivery["webhook"]
        entry["attempts"] += 1
        body = json.dumps(self._payload(delivery, unit, "webhook"), allow_nan=False, default=str).encode("utf-8")
        request = urllib.request.Request(self.webhook, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "User-Agent": "tensorquill-trigger",
            "X-TensorQuill-Event": "incident",
            "X-TensorQuill-Delivery": delivery["id"],
        })
        try:
            with self._opener.open(request, timeout=self.webhook_timeout) as response:
                status = response.status
            problem = None if 200 <= status < 300 else f"the webhook at {self._host} answered HTTP {status}"
        except urllib.error.HTTPError as error:
            redirect = ", a redirect, which is not followed" if 300 <= error.code < 400 else ""
            problem = f"the webhook at {self._host} answered HTTP {error.code}{redirect}"
        except Exception as error:  # refused, timed out, unresolvable, a broken answer
            reason = getattr(error, "reason", None) or error
            problem = f"the webhook at {self._host} could not be reached: {reason}"
        if problem is not None:
            self._failed(delivery, "webhook", problem)
            return False
        entry.update(status="delivered", http_status=status, at_ms=int(time.time() * 1000))
        entry.pop("error", None)
        self._log(f"{self._label(delivery)}: posted to the webhook at {self._host} (HTTP {status})")
        return True

    def _start(self, delivery: dict[str, Any], unit: Mapping[str, Any]) -> None:
        entry = delivery["command"]
        entry["attempts"] += 1
        payload = self._payload(delivery, unit, "command")
        environment = {
            **os.environ,
            "TENSORQUILL_EVIDENCE": payload["evidence"],
            "TENSORQUILL_RUN_ID": payload["run_id"],
            "TENSORQUILL_INCIDENT_ID": payload["id"],
            "TENSORQUILL_INCIDENT_KIND": str(payload["kind"]),
            "TENSORQUILL_INCIDENT_CLASS": str(payload["class"] or ""),
            "TENSORQUILL_INCIDENT_STEP": "" if payload["step"] is None else str(payload["step"]),
            "TENSORQUILL_INCIDENT_RANKS": _rank_ranges(payload["ranks"]),
            "TENSORQUILL_INCIDENT_SUMMARY": str(payload["summary"]).replace("\x00", ""),
        }
        try:
            # From a file, not a pipe: a command that never reads its input must not block the watcher.
            with tempfile.TemporaryFile() as stdin:
                stdin.write(json.dumps(payload, allow_nan=False, default=str).encode("utf-8"))
                stdin.flush()
                stdin.seek(0)
                process = subprocess.Popen(
                    self.command, shell=True, stdin=stdin, env=environment, start_new_session=True
                )
        except (OSError, ValueError) as error:
            self._failed(delivery, "command", f"the command could not be started: {error}")
            return
        # Fired from here on: a watcher restarted while it runs does not start it again.
        entry.update(status="started", pid=process.pid, at_ms=int(time.time() * 1000))
        entry.pop("error", None)
        entry.pop("exit_status", None)
        self._commands.append({"process": process, "delivery": delivery, "since": time.monotonic(), "holds": True})
        self._log(f"{self._label(delivery)}: started the command (pid {process.pid})")

    def _reap(self) -> None:
        """Record how the commands that have ended did, and stop waiting for one that takes too long."""
        still = []
        for item in self._commands:
            status = item["process"].poll()
            delivery = item["delivery"]
            if status is None:
                if item["holds"] and time.monotonic() - item["since"] > self.command_wait:
                    item["holds"] = False
                    self._log(f"{self._label(delivery)}: the command (pid {item['process'].pid}) is still running"
                              f" after {self.command_wait:g} s; it is left running, and the next incident's is no"
                              " longer held back for it")
                still.append(item)
                continue
            entry = delivery["command"]
            entry["exit_status"] = status
            entry.pop("pid", None)
            if status == 0:
                entry["status"] = "done"
            else:
                self._failed(delivery, "command", f"the command exited with status {status}")
        self._commands = still

    # --- The state file --------------------------------------------------------------------

    def _load(self) -> None:
        self._loaded = True
        assert self._state_path is not None
        try:
            text = self._state_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as error:
            self._log(f"trigger: {self._state_path} could not be read ({error}): nothing counts as fired")
            return
        try:
            document = json.loads(text)
            deliveries = document["deliveries"]
            if document.get("version") != STATE_VERSION or not isinstance(deliveries, list):
                raise ValueError("not a version this trigger reads")
            for delivery in deliveries:
                delivery["step"] = _step_key(delivery["step"])
                for rank in delivery["ranks"]:
                    self._fired.add((rank, delivery["kind"], delivery["step"]))
        except (ValueError, KeyError, TypeError) as error:
            self._fired.clear()
            self._log(f"trigger: {self._state_path} is not a state file this version reads ({error}): nothing"
                      " counts as fired, so the incidents listed now fire again")
            return
        self._deliveries = deliveries

    def _save(self) -> None:
        if self._state_path is None:
            return
        document = {"version": STATE_VERSION, "evidence": str(self._evidence_path), "deliveries": self._deliveries}
        text = json.dumps(document, indent=1, default=str) + "\n"
        if text == getattr(self, "_saved", None):
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError as error:
            if not self._save_failed:
                self._save_failed = True
                self._log(f"trigger: {self._state_path} could not be written ({error}): a restarted watcher"
                          " would fire these incidents again")
            return
        self._saved = text
        self._save_failed = False


# --- The command line (`python -m tensorquill`) ----------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--on-incident",
        metavar="COMMAND",
        help="with --watch: a shell command started once for each new incident (new: its rank, kind and step have "
        "not fired before; kept in <evidence>.fired.json beside the evidence, so a restarted watcher does not fire "
        "again). It gets the incident as JSON on standard input and TENSORQUILL_EVIDENCE (the evidence file), "
        "TENSORQUILL_INCIDENT_KIND, _CLASS, _STEP, _RANKS, _SUMMARY, _ID and TENSORQUILL_RUN_ID in its environment; "
        "nothing is added to the command line. It is not waited for or stopped, one runs at a time, and a start "
        "that fails or an exit status other than 0 is tried again on later looks, 4 tries in all. Invariants fire, "
        "and a stalled job or silent rank once the server calls it critical; --fire-anomalies and --fire-kind add "
        "more; a change of run state never fires",
    )
    parser.add_argument(
        "--webhook",
        metavar="URL",
        help="with --watch: an http(s) URL that gets one POST of JSON for each new incident (the same document "
        "the command gets on standard input), with --webhook-timeout seconds, no redirects followed, and the same "
        "tries as --on-incident. Nothing is sent anywhere without this option",
    )
    parser.add_argument(
        "--webhook-timeout", type=float, default=WEBHOOK_TIMEOUT, metavar="SECONDS",
        help=f"seconds one POST to --webhook may take (default: {WEBHOOK_TIMEOUT:g})",
    )
    parser.add_argument(
        "--fire-anomalies",
        action="store_true",
        help="with --on-incident or --webhook: anomalies fire too (statistical judgements against the run's own "
        "history, whose thresholds are not tuned on other jobs yet)",
    )
    parser.add_argument(
        "--fire-kind",
        action="append",
        default=[],
        metavar="KIND",
        help="with --on-incident or --webhook: this kind fires too, whatever its class (repeat the option for more); "
        "the only way an observation fires, but for a stall: job_stalled and rank_silent fire with no option once "
        "their severity is critical, and from their warning when named here (once, not again at critical)",
    )


def from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Trigger | None:
    """The trigger the options ask for; None when neither a command nor a URL was given.
    Options that would be ignored are refused."""
    if args.on_incident is None and args.webhook is None:
        if args.fire_anomalies or args.fire_kind:
            parser.error("--fire-anomalies and --fire-kind choose what fires --on-incident or --webhook: give one of those")
        return None
    if not args.watch:
        parser.error("--on-incident and --webhook fire as incidents appear, so they need --watch")
    try:
        trigger = Trigger(
            args.on_incident, args.webhook, anomalies=args.fire_anomalies, kinds=args.fire_kind,
            webhook_timeout=args.webhook_timeout,
        )
    except ValueError as error:
        parser.error(str(error))
    for kind in args.fire_kind:
        if kind not in INCIDENT_CLASSES:
            print(f"--fire-kind {kind}: not a kind this version knows ({', '.join(sorted(INCIDENT_CLASSES))});"
                  " it fires only if an incident of exactly that kind is listed")
    fires = ["invariants", "a stall at critical", *(["anomalies"] if args.fire_anomalies else []),
             *sorted(set(args.fire_kind))]
    targets = [name for name, given in (("the command", args.on_incident), ("the webhook", args.webhook)) if given]
    print(f"trigger: each new incident fires {' and '.join(targets)} once; firing: {', '.join(fires)}", flush=True)
    return trigger
