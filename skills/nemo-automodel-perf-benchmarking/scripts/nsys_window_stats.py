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

"""Per-microbatch host/GPU accounting from an nsys sqlite export.

Usage: nsys_window_stats.py <report.sqlite> [window_prefix=iteration_13] [layer_class_substring=DecoderLayer]
Analyses the one process that has CUDA activity (nsys_ranks=[0]); NVTX ranges of the other
local processes are ignored. Windows are the `iteration_<n>_ga_step_<k>` NVTX ranges.
"""

import collections
import sqlite3
import sys

db = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else "iteration_13"
layer_class = sys.argv[3] if len(sys.argv) > 3 else "DecoderLayer"
c = sqlite3.connect(db)
ids = {r[0]: r[1] for r in c.execute("select id,value from StringIds")}
pid = c.execute("select globalPid>>24 from CUPTI_ACTIVITY_KIND_KERNEL limit 1").fetchone()[0]
dev = c.execute("select deviceId from CUPTI_ACTIVITY_KIND_KERNEL limit 1").fetchone()[0]


def nvtx_name(r):
    """Return the NVTX range text, resolving a string-table id when the text column is empty."""
    return r[0] if r[0] else ids.get(r[1], "")


wins = [
    (nvtx_name(r), r[2], r[3])
    for r in c.execute(
        "select text,textId,start,end,globalTid from NVTX_EVENTS where end is not null and (globalTid>>24)=? "
        "and (text like ? or textId in (select id from StringIds where value like ?)) order by start",
        (pid, prefix + "_ga_step_%", prefix + "_ga_step_%"),
    )
]
print(f"process {pid} device {dev}; windows: {[(n, round((e - s) / 1e9, 2)) for n, s, e in wins]}")


def category(name):
    """Map a kernel name to a coarse category (gemm, nccl, elementwise, attention, ...)."""
    n = name.lower()
    if "nccl" in n:
        return "nccl"
    if "hybrid_ep" in n or "deep_ep" in n:
        return "hybridep"
    if "flex_attention" in n:
        return "flex_attn"
    if "gemm" in n or "nvjet" in n or "cutlass" in n or "matmul" in n or "xmma" in n:
        return "gemm"
    if "chunk_" in n or "fused_recurrent" in n or "wy_" in n or "solve_tril" in n or "gdn" in n:
        return "fla_gdn"
    if "triton_" in n:
        return "triton_fused_elementwise"
    if "adam" in n or "multi_tensor" in n:
        return "optimizer"
    if "topk" in n or "sort" in n or "radix" in n:
        return "topk_sort"
    if "index" in n or "gather" in n or "scatter" in n or "embedding" in n:
        return "index_gather_scatter"
    if "reduce" in n or "norm" in n or "softmax" in n:
        return "reduce_norm"
    return "elementwise_other"


def union(intervals):
    """Return the total length of the union of the given [start, end) intervals."""
    intervals.sort()
    total = 0
    cur_s = cur_e = None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


agg = collections.defaultdict(float)
n_win = 0
for wname, ws, we in wins:
    n_win += 1
    wall = we - ws
    ks = c.execute(
        "select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=?",
        (dev, ws, we),
    ).fetchall()
    busy = union([(s, e) for s, e, _ in ks])
    agg["wall_ms"] += wall / 1e6
    agg["gpu_busy_ms"] += busy / 1e6
    agg["kernels"] += len(ks)
    cat = collections.defaultdict(float)
    for s, e, nm in ks:
        cat[category(ids.get(nm, ""))] += (e - s) / 1e6
    for k, v in cat.items():
        agg["kcat:" + k] += v
    # GPU idle gaps
    iv = sorted((s, e) for s, e, _ in ks)
    prev = None
    gaps = collections.defaultdict(float)
    for s, e in iv:
        if prev is not None and s > prev:
            g = (s - prev) / 1e3
            b = "<100us" if g < 100 else ("100us-1ms" if g < 1000 else ">1ms")
            gaps[b] += g / 1e3
        prev = max(prev or 0, e)
    for k, v in gaps.items():
        agg["gap:" + k] += v
    api = c.execute(
        "select nameId,start,end from CUPTI_ACTIVITY_KIND_RUNTIME where (globalTid>>24)=? and start>=? and end<=?",
        (pid, ws, we),
    ).fetchall()
    for nid, s, e in api:
        nm = ids.get(nid, "")
        if "LaunchKernel" in nm:
            agg["api_launch_n"] += 1
            agg["api_launch_ms"] += (e - s) / 1e6
        elif "StreamSynchronize" in nm or "DeviceSynchronize" in nm or "EventSynchronize" in nm:
            agg["api_sync_n"] += 1
            agg["api_sync_ms"] += (e - s) / 1e6
        elif "Memcpy" in nm:
            agg["api_memcpy_n"] += 1
        agg["api_total_ms"] += (e - s) / 1e6
    d2h = c.execute(
        "select count(*) from CUPTI_ACTIVITY_KIND_MEMCPY where (globalPid>>24)=? and copyKind=2 and start>=? and end<=?",
        (pid, ws, we),
    ).fetchone()[0]
    agg["memcpy_d2h_n"] += d2h
    # NVTX module ranges of this process inside the window
    rng = c.execute(
        "select text,textId,start,end from NVTX_EVENTS where end is not null and (globalTid>>24)=? and start>=? and end<=? order by start",
        (pid, ws, we),
    ).fetchall()
    mod = collections.defaultdict(lambda: [0, 0.0])
    layer_ranges = []
    for r in rng:
        nm = nvtx_name(r)
        dur = (r[3] - r[2]) / 1e6
        key = nm.split(":")[-1].strip() if ":" in nm else nm
        mod[key][0] += 1
        mod[key][1] += dur
        if layer_class in nm:
            layer_ranges.append((r[2], dur))
    for k, (n, d) in mod.items():
        agg["nvtx_n:" + k] += n
        agg["nvtx_ms:" + k] += d
    # forward vs recompute split of decoder-layer ranges (first N are forward)
    layer_ranges.sort()
    n_layers = len(layer_ranges) // 2
    fwd = sum(d for _, d in layer_ranges[:n_layers]) if layer_ranges else 0.0
    rec = sum(d for _, d in layer_ranges[n_layers:]) if layer_ranges else 0.0
    agg["decoder_layers_fwd_ms"] += fwd
    agg["decoder_layers_recompute_ms"] += rec
    agg["decoder_layer_ranges"] += len(layer_ranges)

if not n_win:
    sys.exit("no windows")
f = lambda k: agg[k] / n_win
print(f"\n== per microbatch (avg over {n_win} windows) ==")
print(
    f"wall {f('wall_ms'):8.0f} ms | GPU busy {f('gpu_busy_ms'):7.0f} ms ({100 * f('gpu_busy_ms') / f('wall_ms'):.1f}%) | kernels {f('kernels'):7.0f}"
)
print(
    f"launch API n={f('api_launch_n'):7.0f} ({f('api_launch_ms'):6.0f} ms) | sync n={f('api_sync_n'):5.0f} ({f('api_sync_ms'):6.0f} ms) | memcpy API n={f('api_memcpy_n'):5.0f} | D2H copies n={f('memcpy_d2h_n'):5.0f} | CUDA API total {f('api_total_ms'):6.0f} ms"
)
print("GPU idle gaps: " + ", ".join(f"{k[4:]} {f(k):.0f} ms" for k in sorted(agg) if k.startswith("gap:")))
print(
    "kernel time by category: "
    + ", ".join(
        f"{k[5:]} {f(k):.0f}"
        for k, _ in sorted(((k, agg[k]) for k in agg if k.startswith("kcat:")), key=lambda x: -x[1])
    )
)
print(
    f"decoder-layer NVTX: ranges {f('decoder_layer_ranges'):.0f}, forward {f('decoder_layers_fwd_ms'):.0f} ms, recompute {f('decoder_layers_recompute_ms'):.0f} ms"
)
print("host NVTX by module (ms, count) top 16:")
tops = sorted(
    ((k[8:], agg[k] / n_win, agg["nvtx_n:" + k[8:]] / n_win) for k in agg if k.startswith("nvtx_ms:")),
    key=lambda x: -x[1],
)
for name, ms, n in tops[:16]:
    print(f"  {ms:8.1f} ms  n={n:6.0f}  {name}")
