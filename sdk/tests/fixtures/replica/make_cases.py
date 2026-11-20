"""Writes the two case files of `REPLICA_OBSERVATION_CONTRACT.md`. Run from the repo root:

    python sdk/tests/fixtures/replica/make_cases.py

The expected results here are written by hand from the contract, never computed: the test
(`sdk/tests/test_replica_contract.py`) reads the rules a second time and must agree with them.
"""
import json
RUN="r1"
A={0:1795000000101,1:1795000000207,2:1795000000350}
# A launch is a nonce rank 0 makes once (contract 2), 32 hex digits; it holds no attempt number.
L0="2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47"; L1="b09d7e6f5a4c4b3a8f2e1d0c9b8a7f65"
def P(norm, s, proj, sig, shape=(8,4), dtype="float32"):
    fp=[norm, s] if proj is None else [norm, s, proj]
    out={"shape": list(shape), "dtype": dtype, "fp": fp}
    if sig is not None: out["sig"]=str(sig)
    return out
W=dict(norm=3.5, s=-1.25, proj=0.75, sig=-4611686018427387904)
B=dict(norm=0.5, s=0.25, proj=-0.125, sig=1152921504606846977, shape=(8,), )
def obs(launch=None, group=(0,1,2), device="cuda", projection="name-v1", signature="fold64-v1", params=None, unread=None, order=("w","b")):
    launch=L0 if launch is None else (None if launch=="unknown" else launch)
    base={"w": P(**W), "b": P(**B)}
    base.update(params or {})
    out={"v":1,"launch":launch,"group":list(group),"device":device,"projection":projection,"signature":signature,
         "params":{k: base[k] for k in order if k in base and base[k] is not None}}
    if unread: out["unread"]=unread
    return out
def rec(rank, step, attempt, **kw):
    r={"schema_version":1,"run_id":RUN,"rank":rank,"attempt":attempt,"step":step}
    if "spread" in kw: r["replica_spread"]=kw.pop("spread")
    if "scope" in kw:  # (launch, group) of an explicit gather: contract 5
        launch, group = kw.pop("scope"); r["replica_spread_scope"]={"launch":launch,"group":list(group)}
    if kw.pop("no_obs", False): return r
    r["replica"]=obs(**kw); return r
def step(launch, group, s, coverage, **kw):
    d={"launch":launch,"group":None if group is None else list(group),"step":s,"coverage":coverage}
    d.update(kw); return d
AGREE=dict(fingerprints="agree", signatures="agree")
def D(s, p, src="replica", launch=L0, group=(0,1)):
    d={"step":s,"parameters":p,"source":src}
    if launch is not None: d.update(launch=launch, group=list(group))
    return d
DR=lambda p: dict(fingerprints="drift",drift_parameters=p,signatures="differ",signature_parameters=p)
NONE=lambda why: dict(fingerprints="not_compared",fingerprints_why=why,signatures="not_compared",signatures_why=why)
CLEAN={"w":[0.0,0.0,0.0],"b":[0.0,0.0,0.0]}
OVER={"w":[0.03,0.01,0.02],"b":[0.0,0.0,0.0]}
WX={"w":P(3.6,-1.2,0.7,999)}  # w far from the other rank's
healthy={
 "healthy_peers_with_distinct_attempts": {"says":"Three ranks of one start, each with its own attempt number. Matched by launch, never by attempt.",
  "records":[rec(r,99,A[r]) for r in (0,1,2)],
  "expected":{"steps":[step(L0,(0,1,2),99,"complete",**AGREE)],"replica_drift":[]}},
 "one_rank_without_a_launch": {"says":"Rank 2's wait for rank 0's launch ran out, so its row names none. The launch's step is insufficient without rank 2, and rank 2's row is said to belong to no known start.",
  "records":[rec(0,99,A[0]),rec(1,99,A[1]),rec(2,99,A[2],launch="unknown")],
  "expected":{"steps":[step(None,(0,1,2),99,"not_established",why="launch"),step(L0,(0,1,2),99,"insufficient",missing=[2])],"replica_drift":[]}},
 "delayed_rank_before_it_arrives": {"says":"Rank 2's observation for step 99 has not arrived. Not clean, not an incident; the missing rank is named.",
  "records":[rec(r,99,A[r]) for r in (0,1)],
  "expected":{"steps":[step(L0,(0,1,2),99,"insufficient",missing=[2])],"replica_drift":[]}},
 "delayed_rank_after_it_arrives": {"says":"The same step once rank 2's observation is there.",
  "records":[rec(r,99,A[r]) for r in (0,1,2)],
  "expected":{"steps":[step(L0,(0,1,2),99,"complete",**AGREE)],"replica_drift":[]}},
 "rank_that_never_reports": {"says":"Rank 2 trains but its telemetry never arrives, on two scheduled steps. Each stays insufficient and says so.",
  "records":[rec(r,s,A[r]) for s in (99,199) for r in (0,1)],
  "expected":{"steps":[step(L0,(0,1,2),99,"insufficient",missing=[2]),step(L0,(0,1,2),199,"insufficient",missing=[2])],"replica_drift":[]}},
 "lone_rank_file": {"says":"One rank's file of a three-rank group.",
  "records":[rec(0,99,A[0])],
  "expected":{"steps":[step(L0,(0,1,2),99,"insufficient",missing=[1,2])],"replica_drift":[]}},
 "stale_rows_after_a_restart": {"says":"The job was started again under the same run ID, by its launcher or by hand (a launcher's restart count would be 0 both times; rank 0's nonce is new each time). The earlier launch reached step 99 on both ranks; the new one has reached it on rank 0 only, with other weights. The new launch's step is insufficient; rank 0's new row is never paired with rank 1's old one.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1)),
             rec(0,99,A[0]+900000,launch=L1,group=(0,1),params={"w":P(3.75,-1.5,0.5,77)})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",**AGREE),step(L1,(0,1),99,"insufficient",missing=[1])],"replica_drift":[]}},
 "two_launches_with_one_attempt_number": {"says":"The job was started again on a host whose clock is behind, and rank 0's attempt number came out the same as in the earlier launch. A launch named by rank 0's attempt would be one launch here, and rank 0's new row would pair with rank 1's old one; the nonce keeps them apart.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1)),
             rec(0,99,A[0],launch=L1,group=(0,1),params={"w":P(3.75,-1.5,0.5,77)})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",**AGREE),step(L1,(0,1),99,"insufficient",missing=[1])],"replica_drift":[]}},
 "launch_not_known": {"says":"The store failed, so no rank knows the launch. The reader cannot say these rows belong to one start, and says that.",
  "records":[rec(r,99,A[r],launch="unknown",group=(0,1)) for r in (0,1)],
  "expected":{"steps":[step(None,(0,1),99,"not_established",why="launch")],"replica_drift":[]}},
 "subgroups": {"says":"Four ranks in two replica groups with different weights. Each group is compared within itself only.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1)),
             rec(2,99,A[2],group=(2,3),params={"w":P(9.0,2.0,1.0,5)}),rec(3,99,A[2]+11,group=(2,3),params={"w":P(9.0,2.0,1.0,5)})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",**AGREE),step(L0,(2,3),99,"complete",**AGREE)],"replica_drift":[]}},
 "reordered_parameters": {"says":"Rank 1 lists the same parameters in another order. Identity is the name, and name-v1 projections do not depend on order.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1),order=("b","w"))],
  "expected":{"steps":[step(L0,(0,1),99,"complete",**AGREE)],"replica_drift":[]}},
 "signature_only_disagreement": {"says":"Fingerprints within the tolerance, signatures different on w. Reported as evidence; no incident under the v1 rule.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1),params={"w":P(3.5,-1.25,0.75,123456789)})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",fingerprints="agree",signatures="differ",signature_parameters=["w"])],"replica_drift":[]}},
 "signature_method_differs": {"says":"Rank 1 records no signature. Fingerprints are compared and agree; signatures are not compared, and are not said to agree.",
  "records":[rec(0,99,A[0],group=(0,1)),
             rec(1,99,A[1],group=(0,1),signature=None,params={"w":P(3.5,-1.25,0.75,None),"b":P(0.5,0.25,-0.125,None,shape=(8,))})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",fingerprints="agree",signatures="not_compared",signatures_why="method")],"replica_drift":[]}},
 "within_tolerance": {"says":"A difference of 1e-9 of the norm on w: below the 1e-6 tolerance.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1),params={"w":P(3.5000000035,-1.25,0.75,W["sig"])})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",**AGREE)],"replica_drift":[]}},
 "unread_on_one_rank": {"says":"Rank 1 could not read b. It is not compared, with why; w is.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1),order=("w",),unread={"b":"on meta"})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",not_compared={"b":"unread"},**AGREE)],"replica_drift":[]}},
 "every_parameter_unread": {"says":"Both ranks reported, and neither could read any parameter. Coverage is complete, since every rank sent a record; nothing was compared, so nothing is said to agree, and the step does not read as checked.",
  "records":[rec(r,99,A[r],group=(0,1),order=(),unread={"w":"on meta","b":"on meta"}) for r in (0,1)],
  "expected":{"steps":[step(L0,(0,1),99,"complete",not_compared={"b":"unread","w":"unread"},**NONE("no_parameters"))],"replica_drift":[]}},
 "group_of_one": {"says":"A replica group of one rank has no rank to be compared with. Its record is complete coverage and nothing is compared.",
  "records":[rec(0,99,A[0],group=(0,))],
  "expected":{"steps":[step(L0,(0,),99,"complete",**NONE("one_rank"))],"replica_drift":[]}},
 "different_device_types": {"says":"One rank on cuda, one on cpu: fingerprints are not compared, and that is said. Signatures are.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1),device="cpu")],
  "expected":{"steps":[step(L0,(0,1),99,"complete",fingerprints="not_compared",fingerprints_why="device",signatures="agree")],"replica_drift":[]}},
 "drift_with_a_rank_missing": {"says":"Ranks 0 and 1 disagree, and rank 2 has not reported. Coverage is insufficient, so nothing is raised yet.",
  "records":[rec(0,99,A[0]),rec(1,99,A[1],params=WX)],
  "expected":{"steps":[step(L0,(0,1,2),99,"insufficient",missing=[2])],"replica_drift":[]}},
 "groups_differ": {"says":"Rank 0 expects a group of two and rank 1 a group of three under one launch: the reader cannot establish who is compared with whom.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1,2),params=WX)],
  "expected":{"steps":[step(L0,None,99,"not_established",why="group")],"replica_drift":[]}},
 "explicit_spread_clean_and_observations_drift": {"says":"An explicit record_replica_spread() at step 99 gathered a clean spread over the same launch, group and parameters, and the observations of the same step disagree on w. The gathered spread decides w: no incident, and the observations' drift is shown as evidence.",
  "records":[rec(0,99,A[0],group=(0,1),spread=CLEAN,scope=(L0,(0,1))),
             rec(1,99,A[1],group=(0,1),params=WX,spread=CLEAN,scope=(L0,(0,1)))],
  "expected":{"steps":[step(L0,(0,1),99,"complete",gathered=["b","w"],**DR(["w"]))],"replica_drift":[]}},
 "zero_norm_parameter": {"says":"A parameter of all zeros on both ranks: the spread's scale is the mean norm, never below 1e-12, so equal zeros agree.",
  "records":[rec(0,99,A[0],group=(0,1),params={"b":P(0.0,0.0,0.0,7,shape=(8,))}),rec(1,99,A[1],group=(0,1),params={"b":P(0.0,0.0,0.0,7,shape=(8,))})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",**AGREE)],"replica_drift":[]}},
 "legacy_replica_spread_below_tolerance": {"says":"A record from before this format: the gathered spread, judged as it always was.",
  "records":[rec(0,99,0,no_obs=True,spread={"w":[0.0,0.0,0.0],"b":[1e-9,0.0,0.0]})],
  "expected":{"steps":[],"legacy":[{"rank":0,"step":99,"drift":False}],"replica_drift":[]}},
}
faulted={
 "drift_on_one_of_three": {"says":"Rank 1's w differs by far more than the tolerance.",
  "records":[rec(0,99,A[0]),rec(1,99,A[1],params=WX),rec(2,99,A[2])],
  "expected":{"steps":[step(L0,(0,1,2),99,"complete",**DR(["w"]))],"replica_drift":[D(99,["w"],group=(0,1,2))]}},
 "drift_between_two_ranks": {"says":"Two ranks disagree on b. Drift is established; which rank is wrong is not (D1: inconclusive).",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1),params={"b":P(0.51,0.25,-0.125,4242,shape=(8,))})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",**DR(["b"]))],"replica_drift":[D(99,["b"])]}},
 "drift_in_the_projection_only": {"says":"Two rows of w swapped on rank 1: norm and sum are equal, the projection and the signature are not.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1),params={"w":P(3.5,-1.25,0.70,31337)})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",**DR(["w"]))],"replica_drift":[D(99,["w"])]}},
 "not_finite_on_one_rank": {"says":"Rank 1's w holds a NaN, written as the string \"nan\". A NaN weight that was read reads as drift.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1),params={"w":P("nan","nan","nan",8)})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",**DR(["w"]))],"replica_drift":[D(99,["w"])]}},
 "shape_differs": {"says":"The two ranks hold w with different shapes: not compared, with why. b drifts and is reported.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1),params={"w":P(3.5,-1.25,0.75,W["sig"],shape=(4,8)),"b":P(0.6,0.25,-0.125,4243,shape=(8,))})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",not_compared={"w":"shape_dtype"},**DR(["b"]))],"replica_drift":[D(99,["b"])]}},
 "projection_setting_differs": {"says":"Rank 1 records no projection. Norm and sum are compared, and b's norm drifts.",
  "records":[rec(0,99,A[0],group=(0,1)),rec(1,99,A[1],group=(0,1),projection=None,params={"w":P(3.5,-1.25,None,W["sig"]),"b":P(0.6,0.25,None,4243,shape=(8,))})],
  "expected":{"steps":[step(L0,(0,1),99,"complete",projection_compared=False,**DR(["b"]))],"replica_drift":[D(99,["b"])]}},
 "explicit_spread_and_observations_on_one_step": {"says":"Both an explicit record_replica_spread() over the same launch and group and the automatic observations at step 99, both showing drift on w. One incident, from the gathered spread.",
  "records":[rec(0,99,A[0],group=(0,1),spread=OVER,scope=(L0,(0,1))),
             rec(1,99,A[1],group=(0,1),params=WX,spread=OVER,scope=(L0,(0,1)))],
  "expected":{"steps":[step(L0,(0,1),99,"complete",gathered=["b","w"],**DR(["w"]))],"replica_drift":[D(99,["w"],"replica_spread")]}},
 "explicit_spread_in_another_launch": {"says":"The earlier launch gathered a clean spread at step 99. The new launch reaches step 99 and its ranks disagree on w. A gathered spread decides only its own launch's step, so the new launch's drift is raised.",
  "records":[rec(0,99,A[0],group=(0,1),spread=CLEAN,scope=(L0,(0,1))),rec(1,99,A[1],group=(0,1),spread=CLEAN,scope=(L0,(0,1))),
             rec(0,99,A[0]+900000,launch=L1,group=(0,1)),rec(1,99,A[1]+900000,launch=L1,group=(0,1),params=WX)],
  "expected":{"steps":[step(L0,(0,1),99,"complete",gathered=["b","w"],**AGREE),step(L1,(0,1),99,"complete",**DR(["w"]))],
              "replica_drift":[D(99,["w"],launch=L1)]}},
 "explicit_spread_of_another_group": {"says":"The job gathered a clean spread over all four ranks at step 99, and the replica group of ranks 0 and 1 disagrees on w. The gathered spread is of another group, so it does not decide this one's step.",
  "records":[rec(0,99,A[0],group=(0,1),spread=CLEAN,scope=(L0,(0,1,2,3))),
             rec(1,99,A[1],group=(0,1),params=WX,spread=CLEAN,scope=(L0,(0,1,2,3)))],
  "expected":{"steps":[step(L0,(0,1),99,"complete",**DR(["w"]))],"replica_drift":[D(99,["w"])]}},
 "explicit_spread_without_a_scope": {"says":"A clean spread table the job computed itself and passed in, so the record cannot say which ranks or which start it covers. It keeps its own verdict and decides nothing else: the observations' drift on w is raised.",
  "records":[rec(0,99,A[0],group=(0,1),spread=CLEAN),rec(1,99,A[1],group=(0,1),params=WX,spread=CLEAN)],
  "expected":{"steps":[step(L0,(0,1),99,"complete",**DR(["w"]))],"replica_drift":[D(99,["w"])]}},
 "explicit_spread_covers_one_parameter": {"says":"The gathered spread of the same launch and group holds w only, and is clean. The observations disagree on w and on b. The gathered spread decides w; b is raised from the observations.",
  "records":[rec(0,99,A[0],group=(0,1),spread={"w":[0.0,0.0,0.0]},scope=(L0,(0,1))),
             rec(1,99,A[1],group=(0,1),params={**WX,"b":P(0.6,0.25,-0.125,4243,shape=(8,))},spread={"w":[0.0,0.0,0.0]},scope=(L0,(0,1)))],
  "expected":{"steps":[step(L0,(0,1),99,"complete",gathered=["w"],**DR(["b","w"]))],"replica_drift":[D(99,["b"])]}},
 "explicit_spread_without_a_scope_and_both_drift": {"says":"A spread with no scope shows drift on w, and so do the observations. They cannot be shown to be the same comparison, so each is listed, with its source. The only case here with two entries.",
  "records":[rec(0,99,A[0],group=(0,1),spread=OVER),rec(1,99,A[1],group=(0,1),params=WX,spread=OVER)],
  "expected":{"steps":[step(L0,(0,1),99,"complete",**DR(["w"]))],"replica_drift":[D(99,["w"]),D(99,["w"],"replica_spread",launch=None)]}},
 "legacy_replica_spread_above_tolerance": {"says":"A record from before this format whose gathered spread is above the tolerance: drift, as it always read.",
  "records":[rec(0,99,0,no_obs=True,spread={"w":[0.02,0.0,0.0],"b":[0.0,0.0,0.0]})],
  "expected":{"steps":[],"legacy":[{"rank":0,"step":99,"drift":True,"parameters":["w"]}],"replica_drift":[D(99,["w"],"replica_spread",launch=None)]}},
}
head={"contract":"REPLICA_OBSERVATION_CONTRACT.md","tolerance":1e-6}
for name,cases in (("no_incident_cases.json",healthy),("incident_cases.json",faulted)):
    json.dump({**head,"cases":cases}, open(f"sdk/tests/fixtures/replica/{name}","w"), indent=1, allow_nan=False)
    print(name, len(cases))
