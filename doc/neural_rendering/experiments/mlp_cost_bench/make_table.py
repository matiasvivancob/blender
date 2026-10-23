#!/usr/bin/env python3
"""Turns results/*.log into the Markdown tables used in the report.

Usage: python3 make_table.py [results_dir]   (writes results/tables.md, prints it)
"""
import glob, os, re, sys

rdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

BUILDS = ["gcc_fast", "gcc_strict", "clang_fast", "gcc_fast_novec", "gcc_cycles", "clang_cycles"]
DESC = {
    "gcc_fast": "g++ -O3 -march=native -ffast-math",
    "gcc_strict": "g++ -O3 -march=native",
    "clang_fast": "clang++ -O3 -march=native -ffast-math",
    "gcc_fast_novec": "g++ -O3 -march=native -ffast-math -fno-tree-vectorize",
    "gcc_cycles": "g++ -O3 -mavx2 -mfma + Cycles math flags (no fast-math)",
    "clang_cycles": "clang++ -O3 -mavx2 -mfma + Cycles math flags (no fast-math)",
}

R = {}      # build -> variant -> dict
PEAK = {}

def parse(fn, into):
    for line in open(fn):
        p = line.strip().split(",")
        if p[0] == "RESULT":
            rec = dict(ns=float(p[3]), worst=float(p[4]), cache=float(p[5]) if float(p[5]) > 0 else None,
                       fmm=float(p[6]), ftot=float(p[7]), cs=float(p[8]))
            old = into.get(p[2])
            if old is None:
                into[p[2]] = rec          # first pass
            else:                          # second pass (analytic only): keep the faster
                if rec["ns"] < old["ns"]:
                    old["ns"] = rec["ns"]
                old["worst"] = max(old["worst"], rec["worst"])
                if rec["cache"] and (old["cache"] is None or rec["cache"] < old["cache"]):
                    old["cache"] = rec["cache"]
        elif p[0] == "PEAK":
            PEAK[b] = dict(vec=float(p[3]), scalar=float(p[5]), ghz=float(p[7]))

for b in BUILDS:
    fn = os.path.join(rdir, b + ".log")
    if not os.path.exists(fn):
        continue
    R[b] = {}
    parse(fn, R[b])
    extra = os.path.join(rdir, b + "_extra.log")
    if os.path.exists(extra):
        parse(extra, R[b])

# GCC's -ffp-contract=on is treated as 'off' (no FMA fusion in plain loops), which halves the
# peak microbenchmark of the gcc_cycles build; use the machine-wide best as the AVX2 peak.
VEC_PEAK = max(v["vec"] for v in PEAK.values()) if PEAK else None
for v in PEAK.values():
    v["vec_machine"] = VEC_PEAK

CFG = [("h16x2", "2x16"), ("h32x2", "2x32"), ("h64x3", "3x64")]
MODES = [("rm", "single, row-major weights (as specified)"),
         ("im", "single, input-major weights (compiler SIMD over outputs)"),
         ("imv", "single, input-major, hand-written float8 + split accumulators"),
         ("b8n", "8-batched SoA, plain loops (compiler SIMD over evals)"),
         ("b8t", "8-batched SoA, plain loops, 8-output register tile"),
         ("b8v", "8-batched SoA, explicit float8, 8-output register tile")]

out = []
def p(s=""):
    out.append(s)

def fmt(x, n=1):
    return "-" if x is None else f"{x:.{n}f}"

def get(b, v):
    return R.get(b, {}).get(v)

def table_main(b, inn=28):
    d = R[b]
    pk = PEAK.get(b)
    p(f"### {b}: `{DESC[b]}`")
    if pk:
        p(f"Peak microbenchmark in this build: {pk['vec']:.1f} GFLOP/s (AVX2 8-wide FMA, 12 chains), {pk['scalar']:.1f} GFLOP/s (scalar FMA). '% of AVX2 peak' uses the machine-wide best, {VEC_PEAK:.1f} GFLOP/s.")
    p()
    p("| config | variant | ns/eval (best of 3) | worst | cache-resident ns | ratio vs analytic | ratio (cache-resident) | matmul FLOPs | GFLOP/s (matmul) | GFLOP/s (all est.) | % of AVX2 peak |")
    p("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    an = d.get("an_ggx_lambert"); anp = d.get("an_ggx_lambert_pdf")
    anb = d.get("an_ggx_lambert_b8"); anv = d.get("an_ggx_lambert_b8v")
    def row(name, desc, r, ref, flops=None):
        if r is None:
            return
        ratio = f"{r['ns']/ref['ns']:.1f}x" if ref else "-"
        ratio_c = f"{r['cache']/ref['cache']:.1f}x" if (ref and r['cache'] and ref['cache']) else "-"
        g = f"{r['fmm']/r['ns']:.1f}" if r['fmm'] else "-"
        gt = f"{r['ftot']/r['ns']:.1f}" if r['ftot'] else "-"
        pc = f"{100*r['fmm']/r['ns']/VEC_PEAK:.0f}%" if (pk and r['fmm']) else "-"
        p(f"| {name} | {desc} | {r['ns']:.1f} | {r['worst']:.1f} | {fmt(r['cache'])} | {ratio} | {ratio_c} | {int(r['fmm']) if r['fmm'] else '-'} | {g} | {gt} | {pc} |")
    row("analytic GGX+Lambert", "single eval, value only", an, an)
    row("analytic GGX+Lambert", "single eval, value + pdf (as Cycles returns)", anp, an)
    row("analytic GGX+Lambert", "8-batched, plain loops", anb, anb)
    row("analytic GGX+Lambert", "8-batched, explicit float8", anv, anv)
    for key, label in CFG:
        for m, mdesc in MODES:
            r = d.get(f"mlp_{key}_in{inn}_{m}")
            batched = m.startswith("b8")
            # ratio: single vs scalar analytic; batched vs batched analytic (explicit float8 one if present)
            ref = (anv or anb) if batched else an
            row(label + f" (IN={inn})", mdesc, r, ref)
    p()

def table_matrix(inn=28):
    builds = [b for b in BUILDS if b in R]
    p(f"### ns/eval, best of 3, IN={inn}, across builds (ratio vs that build's single-eval analytic in parentheses)")
    p()
    p("| variant | " + " | ".join(builds) + " |")
    p("|---|" + "---:|" * len(builds))
    rows = [("an_ggx_lambert", "analytic GGX+Lambert, single"), ("an_ggx_lambert_pdf", "analytic GGX+Lambert+pdf, single"),
            ("an_ggx_lambert_b8", "analytic, 8-batched plain loops"), ("an_ggx_lambert_b8v", "analytic, 8-batched float8")]
    for key, label in CFG:
        for m, _ in MODES:
            rows.append((f"mlp_{key}_in{inn}_{m}", f"{label} {m}"))
    for v, label in rows:
        cells = []
        for b in builds:
            r = R[b].get(v)
            if r is None:
                cells.append("-")
                continue
            an = R[b].get("an_ggx_lambert")
            if v.startswith("mlp") and an:
                ref = an["ns"]
                if "_b8" in v:
                    ref = (R[b].get("an_ggx_lambert_b8v") or R[b].get("an_ggx_lambert_b8") or an)["ns"]
                cells.append(f"{r['ns']:.0f} ({r['ns']/ref:.0f}x)")
            else:
                cells.append(f"{r['ns']:.1f}")
        p(f"| {label} | " + " | ".join(cells) + " |")
    p()

def table_parts():
    b = "gcc_fast"
    if b not in R:
        return
    p("### Cost breakdown pieces (gcc_fast)")
    p()
    p("| piece | ns/eval stream | ns/eval cache-resident |")
    p("|---|---:|---:|")
    for v, label in [("stream_baseline", "memory-only baseline (reads the bytes of one MLP eval)"),
                     ("part_encode12", "encode12 (h, shortest-arc wd')"),
                     ("part_exp3_fast", "3 x Cycles fast_expf"),
                     ("part_exp3_libm", "3 x libm expf"),
                     ("an_ggx", "analytic GGX only"), ("an_lambert", "analytic Lambert only (layered weight)")]:
        r = R[b].get(v)
        if r:
            p(f"| {label} | {r['ns']:.1f} | {fmt(r['cache'])} |")
    p()

for b in BUILDS:
    if b in R:
        table_main(b, 28)
if "gcc_fast" in R:
    p("### gcc_fast, no latent code (IN=12)")
    p()
    d = R["gcc_fast"]
    p("| config | variant | ns/eval | ratio vs analytic | matmul FLOPs | GFLOP/s |")
    p("|---|---|---:|---:|---:|---:|")
    an = d["an_ggx_lambert"]; anv = d.get("an_ggx_lambert_b8v")
    for key, label in CFG:
        for m, mdesc in MODES:
            r = d.get(f"mlp_{key}_in12_{m}")
            if not r:
                continue
            ref = anv if m.startswith("b8") else an
            p(f"| {label} (IN=12) | {m} | {r['ns']:.1f} | {r['ns']/ref['ns']:.1f}x | {int(r['fmm'])} | {r['fmm']/r['ns']:.1f} |")
    p()
table_matrix(28)
table_parts()
txt = "\n".join(out)
open(os.path.join(rdir, "tables.md"), "w").write(txt + "\n")
print(txt)
