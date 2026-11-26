"""The trigger of `--watch`: start a command, post to a webhook, or both, once for each new incident.

    python -m tensorquill run.jsonl --watch --compact \\
        --on-incident 'my-agent "Explain the $TENSORQUILL_INCIDENT_KIND incident in $TENSORQUILL_EVIDENCE"'
    python -m tensorquill store/runs/r --rank all --watch --webhook https://example.com/hooks/tensorquill

**What fires.** An incident is new when its rank, kind and step have not fired before in its run
(the evidence's `run_id`), and a stall when no stall episode of its run is open (below). By
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
to a rank that is silent, unresponsive or lost, or that was one of those when its stream dropped
and is `reconnecting`: a rank in a declared evaluation or checkpoint
phase is `paused`, has none, and so cannot make a stall `critical`. A stall named with
`--fire-kind` fires at `warning` and not again when it becomes `critical`; one not named fires
once, when it becomes `critical`.

**One firing for one stall episode** (the author's decision of Nov 17, 2026, in place of one
firing a run, the same day: a job that recovers and hangs later is a new failure). A stall that
fires opens an episode, kept in the state file with what each stopped rank had stored (its last
step and its attempt) and the run's highest step. No stall fires while it is open. It ends when
a stall is next listed and shows one of:

- progress. For an episode opened by `job_stalled`, the run's step: any rank has stored a step
  beyond the highest one stored when it fired (a lagging rank catching up below that does not
  end it). For one opened by `rank_silent`, the stopped rank's own step: a rank of the episode
  has stored a step beyond the one it stopped at, whether it is still stopped or has recovered
  (a `rank_silent` lists the ranks not stopped under `other_ranks`). The ranks that go on
  stepping beside a silent rank are not in its episode and do not end it, however many looks.
  A rank that stored nothing when it fired ends it with its first stored step;
- a new attempt: the `attempt` the server's status gives a rank of the episode differs from
  the one it had, and no rank of the episode is still stopped exactly as it was (its step and
  attempt), since the ranks of a job started again come back a look or more apart. The server
  learns a new attempt from the process's heartbeats, before it stores any step (and from its
  stored rows, for a sender older than that). A status that gives no attempt says nothing.

Then the next stall at `critical` fires again, under an id of its own. So a silent rank whose
job then stops altogether (`rank_silent`, then `job_stalled`) is one episode. A rank that stops
during an open episode joins it, at the step it stopped at; the episode ends once any rank of it
steps past its own step, so a first rank that recovers while a joined one is still stuck ends it,
and the joined rank's stall then fires. A rank that is only `reconnecting` has not stepped: one
that was stopped when its stream dropped keeps its severity and stays in the stall while it
reconnects and once it is lost (`evidence.rank_is_stopped`), so a hung rank whose connection drops, opens
again under the same attempt and still does not step is one episode and one firing. A hung rank
that is killed and started again comes back as a new attempt, which ends the episode as any new
attempt does: if that one stalls too, it fires again. A
server restart is neither progress nor an attempt: the stall is not listed during the restored
ranks' fresh window, and nothing is decided until it is. (`wandb`'s `run.alert` holds an alert
back by its title and a wait time, not by the step it was raised at.) The status is
the only source: a watcher of a local file given a status with `watchdog=` reads the same
fields from it, and the records' own `attempt` is not used. A state file written before episodes
were kept (keyed by rank, kind and step, or by the run) loads: its last stall is the open
episode, which takes the first stall it sees as its own without firing.

**A lost connection.** A stall's `assessment` says what the server saw: `suspected_hang` when a
stopped rank's stream is still open (it is connected and does not step), `lost_connection` when
a rank's stream dropped and did not come back, as it does when the process is killed (an
out-of-memory kill among them) or the network fails. A killed job keeps firing, labelled so:
the command gets `TENSORQUILL_INCIDENT_ASSESSMENT`, the document gets `assessment`, and the
summary says the connection was lost and that this is not read as a hang.

**What has fired** is kept in a small state file beside the evidence (`run.evidence.json` and
`run.evidence.compact.json` both give `run.evidence.fired.json`), so a watcher that is
restarted does not fire again. What has fired and each open stall episode are kept by run (from
Nov 19, 2026), so a new run whose evidence is written to the same path fires its own incidents,
even one at the same rank, kind and step as the run before. A state file written before that is
read with each firing's run taken from the start of its id (`<run_id>/...`).
Delete the file to start over.

**The compact file.** On the command line a trigger turns on `--compact` (decided Nov 17, 2026),
so the agent that is woken reads the compact evidence: the watcher writes the full file as
before, and the trigger writes `<name>.compact.json` beside it on every look and names that one
in `TENSORQUILL_EVIDENCE` and `evidence`. With `--compact` or `--compact-window` given, the
watcher writes the compact file alone, as it always has, and that is the one named.
`Trigger(compact_window=...)` does the same from Python; it is off by default there.

**The command** is one string, run by the shell. It gets:

- on standard input, one JSON document, the same one the webhook posts (below);
- in its environment, `TENSORQUILL_EVIDENCE` (the evidence file's absolute path),
  `TENSORQUILL_RUN_ID`, `TENSORQUILL_INCIDENT_ID`, `TENSORQUILL_INCIDENT_MCP_ID` (below),
  `TENSORQUILL_INCIDENT_KIND`,
  `TENSORQUILL_INCIDENT_CLASS`, `TENSORQUILL_INCIDENT_STEP`, `TENSORQUILL_INCIDENT_RANKS`
  (`0`, `0-3, 7`, or empty for an incident of the whole run), `TENSORQUILL_INCIDENT_SUMMARY`
  and `TENSORQUILL_INCIDENT_ASSESSMENT` (a stall's `suspected_hang`, `lost_connection`,
  `ranks_never_connected` or `confirmed_hang`; empty for an incident without one).

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

    {"event": "incident", "version": 1, "id": "<run_id>/rank0/non_finite/412",
     "mcp_id": "<run_id>/rank0/non_finite@412", "attempt": 1,
     "sent_at_ms": ..., "run_id": ..., "evidence": "<absolute path>", "run_state": "unhealthy",
     "kind": "non_finite", "class": "invariant", "step": 412, "rank": 0, "ranks": [0],
     "new_ranks": [0], "assessment": null, "summary": "<one line>",
     "incident": {<as in the evidence file>}}

`id` names the delivery, the same on every try, and is what it has always been. `mcp_id` (from
Nov 22, 2026, wave AH; a key added to version 1) names the incident for the MCP server: pass it
as `id` to `get_incident` and it finds this incident of this run and these ranks, whether that
server was started on the run folder, a `rank=<n>` folder, the capture or the evidence file
(`mcp_server.lookup_id`: `<run_id>/<ranks>/<kind>@<step>`, `#2` onwards for a second incident of
that kind and step on those ranks; `<ranks>` is `rank0`, `ranks0-3,7` or `run`). The delivery id
is not a lookup id, and the ids `list_incidents` gives (`<kind>@<step>`) are numbered over the
whole view, so they are not the same from one view to another. It is made from the evidence at
each look and is not kept in the state file; null when none could be made. 

**A stall's `mcp_id`** is of its episode, not of what the stall reads at any moment: it is the
delivery id in the lookup form (`<run_id>/run/rank_silent@49`, and `~2` for a later episode that
fired at the same kind and step). A stall changes its kind and step inside one episode (a
`rank_silent` becomes a `job_stalled`), and a later stall of the same run, or a job started again,
can read exactly as an earlier one did, so the incident alone never says which episode it is. Only
this state file does. `get_incident` therefore resolves a stall's `mcp_id` only on the evidence
file named in `TENSORQUILL_EVIDENCE` (or its compact or full twin), whose state file must show that
firing's episode still open, kept with its baseline, and not ended by the stall listed now
(`episode_ended_by`, the rule above). Then it answers with the run's stall as it reads now, with
`looked_up_as` when the kind or step has changed. In every other case it answers
`original_incident_no_longer_identifiable`, with the reason, and names the stall listed now under
that stall's own id for a separate lookup: it never presents a later stall as the one that
fired.

`rank` is null and `ranks` lists them when several ranks share the incident; `ranks` is empty
for an incident of the whole run. A number that is not finite is the string `"nan"`, `"inf"` or `"-inf"`,
as the capture and the evidence file write it, so the body is strict JSON (it was null before Nov 19, 2026).

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

from .evidence import (
    ALL_RANKS_VIEW, INCIDENT_CLASSES, _group_line, _rank_ranges, incident_class, json_safe, rank_is_stopped,
    replace_atomically,
)

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


def _names_evidence(state_path: Path, evidence: Path) -> bool:
    """Whether the state file at `state_path` says it handed over `evidence`. Its `evidence` is the
    absolute path `_save` wrote, and the state file is beside that file, so the file's name is what
    is compared: a run folder moved or mounted elsewhere with its files still matches."""
    try:
        document = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    named = document.get("evidence") if isinstance(document, dict) else None
    return isinstance(named, str) and Path(named).name == evidence.name


def state_path_handed(evidence_path: str | Path) -> Path:
    """The state file of the trigger that handed over `evidence_path` in `TENSORQUILL_EVIDENCE`,
    or of the full file that one is the compact copy of: the one `tensorquill-mcp` reads.

    The trigger names its state file from the file the watcher wrote (`state_path_for`), and the
    compact copy it hands over is `compact.compact_path` of that file. One strip of `.compact` in
    `state_path_for` gives the same state file for every name but one ending in `.compact.json` or
    `.compact`, or with a suffix other than `.json`: `stall.compact.json` keeps `stall.fired.json`
    and hands `stall.compact.compact.json`, whose own name gives `stall.compact.fired.json` (wave
    AP, AO-06). No state file is renamed: a new name would fire every past incident again. So the
    file's own name (`state_path_for`) stands, unless the full file it is the compact copy of
    (`compact.full_path_of`) has another state file, and that state file says it handed over this
    file or that full one while the file's own does not say it handed this one. A state file
    records the evidence it handed (`_save`), so the two are told apart by what they say, never by
    which of them exists."""
    from .compact import full_path_of

    path = Path(evidence_path)
    own = state_path_for(path)
    full = full_path_of(path)
    if full is None:
        return own
    beside = state_path_for(full)
    if beside == own or _names_evidence(own, path):
        return own
    if _names_evidence(beside, path) or _names_evidence(beside, full):
        return beside
    return own


def _strict(value: Any) -> Any:
    """`value` with every NaN and Inf as the string the capture writes for it (`json_safe`), for a
    body any JSON parser reads."""
    if isinstance(value, float):
        return json_safe(value)
    if isinstance(value, Mapping):
        return {str(k): _strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(v) for v in value]
    return value


def _step_key(step: Any) -> Any:
    return step if step is None or isinstance(step, (int, str)) else str(step)


def _fired_key(run: str | None, rank: Any, kind: Any, step: Any) -> tuple[Any, ...]:
    """What a firing of any incident but a stall holds back: the same kind at the same step on the
    same rank of the same run. A stall is held back by its episode instead (`_follow_episode`).
    The run is part of it (lane U7, Nov 19, 2026): a second run written to the same evidence
    path had its first incident held back by the first run's firing at that rank, kind and step."""
    return (run, rank, kind, step)


def _who_key(ranks: Iterable[Any]) -> str:
    """The ranks part of a delivery id: `rank0`, `ranks0-3,7`, or `run` for the whole run."""
    numbers = [r for r in ranks if isinstance(r, int)]
    return f"rank{'s' if len(numbers) != 1 else ''}{_rank_ranges(numbers).replace(' ', '')}" if numbers else "run"


def _delivery_run(delivery: Mapping[str, Any]) -> str | None:
    """The run a saved firing is of: its `run_id`, or for a state file written before that was
    kept, the start of its id, which is `<run_id>/<ranks>/<kind>/<step>` (and `/<n>` for a later
    stall episode at the same step). None when the id is not one of those."""
    run = delivery.get("run_id")
    if isinstance(run, str):
        return run
    found = delivery.get("id")
    if not isinstance(found, str):
        return None
    tail = f"/{_who_key(delivery.get('ranks') or ())}/{delivery.get('kind')}/{delivery.get('step')}"
    head, slash, number = found.rpartition("/")
    for candidate in (found, head if slash and number.isdecimal() else None):
        if candidate is not None and candidate.endswith(tail) and len(candidate) > len(tail):
            return candidate[: -len(tail)]
    return None


def _number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _stopped(incident: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The stopped ranks of a stall, by rank (as a string, the state file's key), with the last
    step the server stored for each and its attempt (None when the status does not say)."""
    found = {}
    for entry in incident.get("ranks") or ():
        if rank_is_stopped(entry):
            found[str(entry.get("rank"))] = {"last_step": _number(entry.get("last_step")),
                                             "attempt": entry.get("attempt")}
    return found


def _others(incident: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The ranks a `rank_silent` lists as not stopped (`other_ranks`), as `_stopped` gives them."""
    return {str(e.get("rank")): {"last_step": _number(e.get("last_step")), "attempt": e.get("attempt")}
            for e in incident.get("other_ranks") or () if isinstance(e, Mapping)}


def _beyond(now: int | None, before: int | None) -> bool:
    """A stored step beyond `before`; a baseline of None (nothing stored then) is below any step."""
    return now is not None and (before is None or now > before)


def episode_ended_by(episode: Mapping[str, Any], stall: Mapping[str, Any]) -> str | None:
    """Whether `stall`, as it is listed now, ends the open `episode`: `"attempt"` when the job was
    started again, `"progress"` when a step was stored beyond where the episode stopped, None while
    the episode goes on. The one rule, of the trigger (`Trigger._follow_episode`) and of the MCP
    server when it asks whether the stall listed now is still the one a firing was for
    (`mcp_server`, a stall's lookup id). It changes nothing."""
    members = episode.get("ranks") or {}
    stopped = _stopped(stall)
    # Every rank the stall says something of: stopped, and (for `rank_silent`) the others, so
    # a rank of the episode that has recovered is seen stepping on.
    seen = {**_others(stall), **stopped}
    # A rank of the episode still stopped exactly as it was (its step and attempt): the job
    # is not started again for it yet, as when ranks of a restarted job start a look apart.
    unchanged = any(rank in members and members[rank].get("last_step") == now["last_step"]
                    and members[rank].get("attempt") == now["attempt"] for rank, now in stopped.items())
    for rank, now in seen.items():
        before = members.get(rank)
        if not unchanged and before is not None and now["attempt"] is not None \
                and before.get("attempt") is not None and now["attempt"] != before["attempt"]:
            return "attempt"  # a new attempt: the job was started again
    if episode.get("kind") == "job_stalled":
        # The run's step: the stall is over once any rank has stored a step beyond the
        # highest one stored when it fired. Of the ranks the stall lists, or, read from the
        # stall's own step, of a rank that has since finished and is no longer listed. An
        # episode from a state file written before `stall_step` was kept has only the first.
        opened_at = _number(episode.get("stall_step"))
        if _beyond(_run_step(stall), episode.get("run_step")) or (
                stall.get("kind") == "job_stalled" and opened_at is not None
                and _beyond(_number(stall.get("step")), opened_at)):
            return "progress"
    else:
        # `rank_silent`: each stopped rank's own step. The ranks that go on stepping beside
        # it are not in the episode and do not end it.
        for rank, now in seen.items():
            if rank in members and _beyond(now["last_step"], members[rank].get("last_step")):
                return "progress"
    return None


def stall_lookup_id(delivery: Mapping[str, Any]) -> str | None:
    """A stall firing's `mcp_id`: its delivery id in the lookup form, `<run_id>/<ranks>/<kind>@<step>`,
    and `~2` onwards for a later stall episode that fired at the same kind and step (the delivery
    id's `/2`). One id for one episode, whatever the stall reads later: the MCP server finds the
    episode by it in the state file. None for a delivery whose id is not of that shape."""
    run, found = _delivery_run(delivery), delivery.get("id")
    if run is None or not isinstance(found, str):
        return None
    who, kind, step = _who_key(delivery.get("ranks") or ()), delivery.get("kind"), delivery.get("step")
    base = f"{run}/{who}/{kind}/{step}"
    if found == base:
        number = 1
    elif found.startswith(base + "/") and found[len(base) + 1:].isdecimal():
        number = int(found[len(base) + 1:])
    else:
        return None
    named = f"{kind}@{step}" if isinstance(step, int) and not isinstance(step, bool) else str(kind)
    return f"{run}/{who}/{named}" + (f"~{number}" if number > 1 else "")


def _run_step(incident: Mapping[str, Any]) -> int | None:
    """The run's step in a stall: the highest step any rank has stored."""
    steps = [_number(e.get("last_step")) for e in incident.get("ranks") or () if isinstance(e, Mapping)]
    steps.append(_number(incident.get("newest_step")))
    steps = [step for step in steps if step is not None]
    return max(steps) if steps else None


def _who(ranks: Iterable[Any]) -> str:
    numbers = [r for r in ranks if isinstance(r, int)]
    if not numbers:
        return "the run"
    return f"rank{'s' if len(numbers) != 1 else ''} {_rank_ranges(numbers, 8)}"


class Trigger:
    """Fires once for each new incident of the evidence a watcher writes (see the module's text).

    `watch(..., trigger=Trigger("my-command"))` calls `tick()` on every look and
    `look(evidence, path)` after every write; neither ever raises. Invariants fire, and a
    stall at `critical` (`STALL_KINDS`), once an episode; `anomalies` adds the anomalies, `kinds` any
    kinds by name, a stall at `warning` among them. `state_path` replaces the state file beside
    the evidence, and is read when the trigger is made. `compact_window` writes the compact
    evidence beside the full file at each look and hands that one over (the command line sets
    it). `log` gets each line this prints.
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
        compact_window: int | None = None,
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
        self._fired: set[tuple[Any, ...]] = set()
        # The stall episode that has fired and not ended (see the module's text), for each run, as
        # the state file keeps them; a run with none may fire a stall. `_episode` is the run's
        # the watcher looks at now.
        self._episodes: dict[str, dict[str, Any]] = {}
        self._stalls: list[Mapping[str, Any]] = []
        # Set: the evidence written is the full file, and this writes the compact one beside it.
        self.compact_window = compact_window
        self._compact_failed = False
        self._units: list[dict[str, Any]] = []
        self._evidence_path: Path | None = None
        self._run_id = ""
        self._run_state: Any = None
        self._commands: list[dict[str, Any]] = []
        self._retry_at: dict[tuple[str, str], float] = {}
        self._unknown: set[str] = set()
        self._save_failed = False
        if self._state_path is not None:
            # Read now (DET-01, Nov 18, 2026): both watchers tick before their first look, and a
            # file read only at that look was first written over with nothing fired.
            try:
                self._load()
            except Exception as error:  # a state file that cannot be read never stops the watcher
                self._fired.clear()
                self._episodes, self._deliveries = {}, []
                self._log(f"trigger: {self._state_path} could not be read ({type(error).__name__}: {error}):"
                          " nothing counts as fired")

    @property
    def _episode(self) -> dict[str, Any] | None:
        return self._episodes.get(self._run_id)

    @_episode.setter
    def _episode(self, episode: dict[str, Any] | None) -> None:
        if episode is None:
            self._episodes.pop(self._run_id, None)
        else:
            self._episodes[self._run_id] = episode

    def _was_fired(self, rank: Any, kind: Any, step: Any) -> bool:
        """This run's incident at that rank, kind and step has fired; or one from a state file
        whose firing names no run that can be read, as such a file held it back before."""
        return (_fired_key(self._run_id, rank, kind, step) in self._fired
                or _fired_key(None, rank, kind, step) in self._fired)

    # --- What the watcher calls ------------------------------------------------------------

    def look(self, evidence: Mapping[str, Any], path: str | Path, *, rank: int | None = None) -> None:
        """Take the evidence just written to `path`: fire for what is new in it. `rank` is the
        rank a one-rank evidence is of (the all-ranks view names its own). Never raises."""
        try:
            self._evidence_path = Path(path).resolve()
            if not self._loaded:
                if self._state_path is None:
                    # From the file the watcher wrote, so the name is the one it always was.
                    self._state_path = state_path_for(self._evidence_path)
                self._load()
            if self.compact_window is not None:
                self._evidence_path = self._write_compact(evidence, self._evidence_path)
            self._run_id = str(evidence.get("run_id") or self._evidence_path.name)
            found = evidence.get("run_state")
            self._run_state = found.get("state") if isinstance(found, Mapping) else None
            if rank is None and isinstance(evidence.get("detection"), Mapping):
                rank = evidence["detection"].get("rank")
            self._units = self._collect(evidence, rank)
            self._name_for_lookup(evidence)
            self._enqueue()
            self._work()
        except Exception as error:  # a trigger never stops the watcher
            self._log(f"trigger: {type(error).__name__}: {error}; the watcher goes on")

    def _write_compact(self, evidence: Mapping[str, Any], path: Path) -> Path:
        """The compact evidence, written beside the full file at `path`; `path` itself when it
        cannot be written, which is said once."""
        from .compact import write_compact

        try:
            written = write_compact(evidence, path, window=self.compact_window).resolve()
        except Exception as error:
            if not self._compact_failed:
                self._log(f"trigger: the compact evidence beside {path} could not be written"
                          f" ({type(error).__name__}: {error}); the full file is handed over instead")
            self._compact_failed = True
            return path
        if self._compact_failed:
            self._log(f"trigger: the compact evidence is written again: {written}")
        self._compact_failed = False
        return written

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
                "ranks": ranks, "incident": incident, "summary": summary,
                "assessment": incident.get("assessment")}

    def _collect(self, evidence: Mapping[str, Any], rank: int | None) -> list[dict[str, Any]]:
        """The incidents of `evidence` that the options let fire, each with the ranks it is on."""
        units = []
        self._stalls = []
        if evidence.get("view") == ALL_RANKS_VIEW:
            for group in evidence.get("incident_groups") or ():
                if isinstance(group, Mapping) and isinstance(group.get("incident"), Mapping):
                    units.append(self._unit(group["incident"], list(group.get("ranks") or ()), group))
            shared = evidence.get("run_incidents") or ()
        else:
            shared = evidence.get("incidents") or ()
        for incident in shared:
            if isinstance(incident, Mapping) and incident.get("kind") in STALL_KINDS:
                self._stalls.append(incident)  # listed, whether or not it fires
            if isinstance(incident, Mapping):
                units.append(self._unit(incident, [rank], {
                    "incident": incident, "kind": incident.get("kind"), "step": incident.get("step"),
                    "ranks": [rank], "incident_rank": rank, "numbers_differ": False,
                }))
        return [unit for unit in units if unit is not None]

    def _name_for_lookup(self, evidence: Mapping[str, Any]) -> None:
        """Each unit's `mcp_id`: the id the MCP server's `get_incident` takes for this incident of
        this run and these ranks (`mcp_server.lookup_id`; wave AH, P5-1). Made at every look from
        the evidence looked at, so a firing still to be sent names its incident as it reads now,
        and never kept in the state file. A unit it cannot be made for has none."""
        try:
            from .mcp_server import lookup_ids

            made = lookup_ids(evidence, self._run_id,
                              [(unit["incident"], [r for r in unit["ranks"] if isinstance(r, int)]) for unit in self._units])
        except Exception as error:  # never the watcher's problem, and never a firing's
            self._log(f"trigger: no MCP lookup id could be made ({type(error).__name__}: {error}); the firings go on without one")
            made = [None] * len(self._units)
        for unit, name in zip(self._units, made):
            unit["mcp_id"] = name

    def _enqueue(self) -> None:
        """A firing for each incident with a rank that has not fired before, for those ranks;
        for a stall, once an episode (`_follow_episode`)."""
        targets = [name for name, given in (("command", self.command), ("webhook", self.webhook)) if given]
        for stall in self._stalls:
            self._follow_episode(stall)
        for unit in self._units:
            if unit["kind"] in STALL_KINDS:
                if self._episode is not None:
                    continue  # the open episode's stall: fired already
                new = list(unit["ranks"])
            else:
                new = [r for r in unit["ranks"] if not self._was_fired(r, unit["kind"], unit["step"])]
                if not new:
                    continue
            delivery_id = f"{self._run_id}/{_who_key(new)}/{unit['kind']}/{unit['step']}"
            taken = {d["id"] for d in self._deliveries}
            if delivery_id in taken:  # a later stall episode at the same step: its own id, for receivers that drop repeats
                delivery_id = next(f"{delivery_id}/{n}" for n in range(2, len(taken) + 3) if f"{delivery_id}/{n}" not in taken)
            self._deliveries.append({
                "id": delivery_id, "run_id": self._run_id,
                "kind": unit["kind"], "step": unit["step"], "ranks": new,
                **({"assessment": unit["assessment"]} if unit["assessment"] is not None else {}),
                "first_seen_ms": int(time.time() * 1000),
                **{target: {"status": "pending", "attempts": 0} for target in targets},
            })
            if unit["kind"] in STALL_KINDS:
                self._episode = self._open_episode(unit["incident"], delivery_id)
            else:
                self._fired.update(_fired_key(self._run_id, r, unit["kind"], unit["step"]) for r in new)
        self._save()

    # --- Stall episodes ----------------------------------------------------------------------

    @staticmethod
    def _open_episode(incident: Mapping[str, Any], delivery_id: str | None) -> dict[str, Any]:
        """The episode a stall that fires opens: what it was stopped at, to tell progress from.
        A `job_stalled` also keeps its own `step` (`stall_step`), the highest step any rank had
        stored, finished ranks included: those are not in its `ranks`, so a rank that steps on
        and then finishes is seen only there (DET-07, lane V5, Nov 19, 2026)."""
        episode = {"kind": incident.get("kind"), "delivery": delivery_id, "run_step": _run_step(incident),
                   "ranks": _stopped(incident)}
        if incident.get("kind") == "job_stalled":
            episode["stall_step"] = _number(incident.get("step"))
        return episode

    def _follow_episode(self, stall: Mapping[str, Any]) -> None:
        """End the open episode if `stall` shows progress or a new attempt since it opened, or
        else take the ranks that have stopped since into it. Decided only when a stall is listed:
        a stall that is not listed for a while (as for the window after a server restart) has
        not ended for that."""
        episode = self._episode
        if episode is None:
            return
        stopped = _stopped(stall)
        if episode.get("legacy"):
            # From a state file that kept no baseline: the first stall seen is taken as the
            # episode's, without firing (a stall listed in the file has fired).
            self._episode = {**self._open_episode(stall, episode.get("delivery")), "adopted": True}
            return
        if episode_ended_by(episode, stall) is not None:
            self._episode = None
            return
        members = episode.setdefault("ranks", {})
        for rank, now in stopped.items():
            members.setdefault(rank, now)  # a rank that stopped during the episode joins it

    def _unit_for(self, delivery: Mapping[str, Any]) -> dict[str, Any] | None:
        """The incident a firing is for, in the evidence last looked at; None once a resumed
        run has rolled it back, and then nothing is sent for it. A stall's firing is for its
        episode's stall as it reads now: a try still to make for a `rank_silent` that has become a
        `job_stalled` sends the `job_stalled` (under the firing's own id). A firing of an episode
        that is over sends nothing more. A firing of another run than the one looked at sends
        nothing either: its incident is not in this evidence, whatever this run's matches it."""
        if _delivery_run(delivery) not in (self._run_id, None):
            return None
        if delivery["kind"] in STALL_KINDS:
            # Only the open episode's firing has anything to send: an earlier episode is over.
            if self._episode is None or self._episode.get("delivery") != delivery["id"]:
                return None
            return next((unit for unit in self._units if unit["kind"] in STALL_KINDS), None)
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
        lost = " (lost connection)" if delivery.get("assessment") == "lost_connection" else ""
        return f"trigger: {delivery['kind']}{lost} at step {delivery['step']} ({_who(delivery['ranks'])})"

    def _payload(self, delivery: Mapping[str, Any], unit: Mapping[str, Any], target: str) -> dict[str, Any]:
        ranks = [r for r in unit["ranks"] if isinstance(r, int)]
        return _strict({
            "event": "incident",
            "version": PAYLOAD_VERSION,
            "id": delivery["id"],
            # A stall's is of its firing, not of what the stall reads now: one id for one episode.
            "mcp_id": stall_lookup_id(delivery) if delivery.get("kind") in STALL_KINDS else unit.get("mcp_id"),
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
            "assessment": unit["assessment"],
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
            "TENSORQUILL_INCIDENT_MCP_ID": str(payload["mcp_id"] or ""),
            "TENSORQUILL_INCIDENT_KIND": str(payload["kind"]),
            "TENSORQUILL_INCIDENT_CLASS": str(payload["class"] or ""),
            "TENSORQUILL_INCIDENT_STEP": "" if payload["step"] is None else str(payload["step"]),
            "TENSORQUILL_INCIDENT_RANKS": _rank_ranges(payload["ranks"]),
            "TENSORQUILL_INCIDENT_SUMMARY": str(payload["summary"]).replace("\x00", ""),
            "TENSORQUILL_INCIDENT_ASSESSMENT": str(payload["assessment"] or ""),
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
                if delivery["kind"] in STALL_KINDS:
                    continue
                run = _delivery_run(delivery)  # a file written before the run was kept: from the id
                for rank in delivery["ranks"]:
                    self._fired.add(_fired_key(run, rank, delivery["kind"], delivery["step"]))
            by_id = {d.get("id"): d for d in deliveries}

            def run_of(episode: Mapping[str, Any]) -> str | None:
                return _delivery_run(by_id[episode["delivery"]])

            def readable(episode: Any) -> bool:
                return isinstance(episode, dict) and isinstance(episode.get("ranks", {}), dict) \
                    and episode.get("delivery") in by_id and run_of(episode) is not None

            episodes = document.get("episodes")
            episode = document.get("episode")
            stalls = [d for d in deliveries if d["kind"] in STALL_KINDS]
            if isinstance(episodes, dict):
                # Each run's open episode (from Nov 19, 2026), filed under the run its firing is of.
                for found in episodes.values():
                    if readable(found):
                        self._episodes[run_of(found)] = found
                    else:
                        self._log(f"trigger: {self._state_path}: the stall episode it lists has no firing in it, or"
                                  " cannot be read; a stall may fire again")
            elif readable(episode):
                # One episode, written before the run was kept: the run of its firing's id.
                self._episodes[run_of(episode)] = episode
            elif episode is not None:
                # Its firing was removed from the file (to try a given-up stall again), or it is
                # not an episode this version reads: no stall episode is open.
                self._log(f"trigger: {self._state_path}: the stall episode it lists has no firing in it, or cannot be"
                          " read; a stall may fire again")
            elif "episode" not in document and stalls:
                # Written before episodes were kept (keyed by rank, kind and step, or by the run):
                # its last stall is the open episode of that stall's run, with no baseline yet.
                legacy = {"legacy": True, "delivery": stalls[-1]["id"], "kind": stalls[-1]["kind"]}
                self._episodes[_delivery_run(stalls[-1]) or ""] = legacy
        except (ValueError, KeyError, TypeError) as error:
            self._fired.clear()
            self._episodes = {}
            self._log(f"trigger: {self._state_path} is not a state file this version reads ({error}): nothing"
                      " counts as fired, so the incidents listed now fire again")
            return
        self._deliveries = deliveries

    def _save(self) -> None:
        # Nothing is written before the first look: until then nothing has fired, been tried or
        # ended, and the file read at the start (a `state_path` given) is left as it was.
        if self._state_path is None or self._evidence_path is None:
            return
        # `episodes`: each run's open episode, by run. `episode`, the one of the run looked at now,
        # is still written for a trigger older than Nov 19, 2026 that reads the same version.
        document = {"version": STATE_VERSION, "evidence": str(self._evidence_path), "deliveries": self._deliveries,
                    "episode": self._episode, "episodes": self._episodes}
        text = json.dumps(json_safe(document), indent=1, default=str, allow_nan=False) + "\n"
        if text == getattr(self, "_saved", None):
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            # A temporary file of this writer's own: two watchers given one state path no longer
            # take each other's `.tmp` (lane U7, Nov 19, 2026). The file still has one writer at
            # a time in mind: the later save wins.
            replace_atomically(self._state_path, text)
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
        "not fired before in its run, and for a stall, once an episode: until the run steps past it or a new attempt; kept in <evidence>.fired.json beside the evidence, so a "
        "restarted watcher does not fire again). Turns on --compact: the full evidence is written as before and "
        "the compact file beside it is the one handed over. It gets the incident as JSON on standard input and "
        "TENSORQUILL_EVIDENCE (the compact evidence file), TENSORQUILL_INCIDENT_KIND, _CLASS, _STEP, _RANKS, "
        "_SUMMARY, _ASSESSMENT, _ID and TENSORQUILL_RUN_ID in its environment; "
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
        "tries as --on-incident. Turns on --compact, as --on-incident does. Nothing is sent anywhere without this "
        "option",
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
        "their severity is critical, and from their warning when named here (once an episode, not again at critical)",
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
    # The agent that is woken reads the compact evidence: with no --compact the watcher writes the
    # full file as before and the trigger writes the compact one beside it.
    from .compact import DEFAULT_WINDOW

    beside = DEFAULT_WINDOW if getattr(args, "compact_window", None) is None else None
    try:
        trigger = Trigger(
            args.on_incident, args.webhook, anomalies=args.fire_anomalies, kinds=args.fire_kind,
            webhook_timeout=args.webhook_timeout, compact_window=beside,
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
    if beside is not None:
        print("trigger: hands over the compact evidence (<evidence>.compact.json), written beside the full file",
              flush=True)
    return trigger
