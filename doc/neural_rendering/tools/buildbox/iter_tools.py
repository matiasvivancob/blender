#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Small helpers for the iteration-time measurement in README.md, so the same
commands work in bash, cmd and PowerShell (no sed, no Set-Content that would
change encoding or line endings of a source file).

    python iter_tools.py edit add    intern/cycles/kernel/closure/bsdf.h
    python iter_tools.py edit remove intern/cycles/kernel/closure/bsdf.h
    python iter_tools.py ninjalog <build dir> [-n 10]
    python iter_tools.py depcheck <build dir> [--filter intern/cycles] [--max-zero 0.1]

`edit add` appends one comment line `/* iteration-timing <nanoseconds> */` (a real
content change, so a compiler cache cannot hit), keeping the file's own line
ending style. `edit remove` deletes exactly the lines carrying that marker, and
nothing else, so unrelated uncommitted edits in the same file survive (do not
use `git checkout -- <file>` for this). `ninjalog` prints the slowest steps from
`<build dir>/.ninja_log`. `depcheck` runs `ninja -t deps` and exits 1 when more than
`--max-zero` of the matching objects have no recorded header dependencies (the
silent stale-build failure of README section 5, Step 3); run it before kernel work
and after any reconfigure.
"""

import argparse
import os
import sys
import time

MARKER = b"iteration-timing"


def edit_add(path):
    with open(path, "rb") as f:
        data = f.read()
    if MARKER in data:
        print("%s already has an iteration-timing line; run `edit remove` first" % path, file=sys.stderr)
        return 1
    eol = b"\r\n" if b"\r\n" in data else b"\n"
    if data and not data.endswith(b"\n"):
        print("%s does not end with a newline; refusing (remove could not restore it exactly)" % path, file=sys.stderr)
        return 1
    with open(path, "wb") as f:
        f.write(data + b"/* " + MARKER + b" " + str(time.time_ns()).encode() + b" */" + eol)
    print("added marker line to %s" % path)
    return 0


def edit_remove(path):
    with open(path, "rb") as f:
        lines = f.read().splitlines(keepends=True)
    kept = [l for l in lines if MARKER not in l]
    if len(kept) == len(lines):
        print("no iteration-timing line in %s" % path)
        return 0
    with open(path, "wb") as f:
        f.write(b"".join(kept))
    print("removed %d marker line(s) from %s" % (len(lines) - len(kept), path))
    return 0


def ninjalog(build_dir, top):
    path = os.path.join(build_dir, ".ninja_log")
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            start, end, out = int(parts[0]), int(parts[1]), parts[3]
            rows.append(((end - start) / 1000.0, out))
    for secs, out in sorted(rows, reverse=True)[:top]:
        print("%8.1f s  %s" % (secs, out))
    return 0


def parse_deps(text, needle):
    """(objects, objects with zero deps) from `ninja -t deps` output, for targets containing needle."""
    import re
    total, zero = [], []
    for line in text.splitlines():
        m = re.match(r"^(\S.*?): #deps (\d+)", line)
        if not m:
            continue
        target = m.group(1).replace("\\", "/")
        if needle and needle not in target:
            continue
        if not target.endswith((".obj", ".o")):
            continue
        total.append(target)
        if m.group(2) == "0":
            zero.append(target)
    return total, zero


def depcheck(build_dir, needle, max_zero):
    import subprocess
    try:
        out = subprocess.run(["ninja", "-C", build_dir, "-t", "deps"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True, encoding="utf-8",
                             errors="replace", check=True).stdout
    except (OSError, subprocess.CalledProcessError) as ex:
        print("depcheck: could not run ninja -t deps: %s" % ex, file=sys.stderr)
        return 2
    total, zero = parse_deps(out, needle)
    if not total:
        print("depcheck FAIL: no built objects matching %r; build first" % needle)
        return 1
    frac = len(zero) / len(total)
    print("depcheck: %d of %d objects matching %r have no header dependencies (%.1f%%, limit %.1f%%)"
          % (len(zero), len(total), needle, 100 * frac, 100 * max_zero))
    for t in zero[:5]:
        print("  #deps 0: " + t)
    if frac > max_zero:
        print("depcheck FAIL: header edits will not rebuild these objects; see README section 5, Step 3")
        return 1
    print("depcheck PASS")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("edit")
    e.add_argument("action", choices=("add", "remove"))
    e.add_argument("path")
    n = sub.add_parser("ninjalog")
    n.add_argument("build_dir")
    n.add_argument("-n", type=int, default=10)
    d = sub.add_parser("depcheck")
    d.add_argument("build_dir")
    d.add_argument("--filter", default="intern/cycles")
    d.add_argument("--max-zero", type=float, default=0.1)
    a = p.parse_args(argv)
    if a.cmd == "depcheck":
        return depcheck(a.build_dir, a.filter, a.max_zero)
    if a.cmd == "edit":
        return edit_add(a.path) if a.action == "add" else edit_remove(a.path)
    return ninjalog(a.build_dir, a.n)


if __name__ == "__main__":
    sys.exit(main())
