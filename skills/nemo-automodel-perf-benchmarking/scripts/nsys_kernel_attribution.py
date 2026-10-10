# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attribute every kernel in one steady microbatch to the innermost NVTX module range (forward vs backward).

Usage: nsys_kernel_attribution.py <report.sqlite> [window=iteration_13_ga_step_0] [top=14]
"""

import bisect
import collections
import sqlite3
import sys

db = sys.argv[1]
window = sys.argv[2] if len(sys.argv) > 2 else "iteration_13_ga_step_0"
top = int(sys.argv[3]) if len(sys.argv) > 3 else 14
c = sqlite3.connect(db)
ids = {r[0]: r[1] for r in c.execute("select id,value from StringIds")}
pid = c.execute("select globalPid>>24 from CUPTI_ACTIVITY_KIND_KERNEL limit 1").fetchone()[0]
dev = c.execute("select deviceId from CUPTI_ACTIVITY_KIND_KERNEL limit 1").fetchone()[0]
ws, we = c.execute(
    "select start,end from NVTX_EVENTS where end is not null and (globalTid>>24)=? and (text=? or textId in (select id from StringIds where value=?))",
    (pid, window, window),
).fetchone()
fs, fe = c.execute(
    "select start,end from NVTX_EVENTS where end is not null and (globalTid>>24)=? and start>=? and end<=? and (text like '%ForConditionalGeneration%' or textId in (select id from StringIds where value like '%ForConditionalGeneration%')) order by start limit 1",
    (pid, ws, we),
).fetchone()
api = {
    cid: s
    for cid, s in c.execute(
        "select correlationId,start from CUPTI_ACTIVITY_KIND_RUNTIME where (globalTid>>24)=? and start>=? and end<=? and nameId in (select id from StringIds where value like '%LaunchKernel%')",
        (pid, ws, we),
    )
}
rng = sorted(
    (s, e, (t if t else ids.get(tid, "")))
    for t, tid, s, e in c.execute(
        "select text,textId,start,end from NVTX_EVENTS where end is not null and (globalTid>>24)=? and start>=? and end<=?",
        (pid, ws, we),
    )
)
starts = [r[0] for r in rng]


def module_of(t):
    """Return the innermost NVTX module range that contains a kernel launch timestamp."""
    i = bisect.bisect_right(starts, t)
    best = None
    for j in range(i - 1, max(i - 400, -1), -1):
        s, e, nm = rng[j]
        if s <= t <= e and (best is None or (e - s) < (best[1] - best[0])):
            best = (s, e, nm)
    return best[2].split(":")[-1].strip() if best else "?"


fwd, bwd, fwd_t, bwd_t = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
for s, e, cid in c.execute(
    "select start,end,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=?",
    (dev, ws, we),
):
    if cid not in api:
        continue
    key = module_of(api[cid])
    if api[cid] < fe:
        fwd[key] += 1
        fwd_t[key] += (e - s) / 1e6
    else:
        bwd[key] += 1
        bwd_t[key] += (e - s) / 1e6
print(f"window {window}: forward kernels {sum(fwd.values())}, backward(+recompute) kernels {sum(bwd.values())}")
print("FORWARD by innermost module (count, GPU ms):")
for k, n in fwd.most_common(top):
    print(f"  {n:6d}  {fwd_t[k]:6.0f} ms  {k}")
print("BACKWARD(+recompute) by innermost module:")
for k, n in bwd.most_common(top):
    print(f"  {n:6d}  {bwd_t[k]:6.0f} ms  {k}")
