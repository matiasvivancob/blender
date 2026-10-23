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

`edit add` appends one comment line `/* iteration-timing <nanoseconds> */` (a real
content change, so a compiler cache cannot hit), keeping the file's own line
ending style. `edit remove` deletes exactly the lines carrying that marker, and
nothing else, so unrelated uncommitted edits in the same file survive (do not
use `git checkout -- <file>` for this). `ninjalog` prints the slowest steps from
`<build dir>/.ninja_log`.
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("edit")
    e.add_argument("action", choices=("add", "remove"))
    e.add_argument("path")
    n = sub.add_parser("ninjalog")
    n.add_argument("build_dir")
    n.add_argument("-n", type=int, default=10)
    a = p.parse_args(argv)
    if a.cmd == "edit":
        return edit_add(a.path) if a.action == "add" else edit_remove(a.path)
    return ninjalog(a.build_dir, a.n)


if __name__ == "__main__":
    sys.exit(main())
