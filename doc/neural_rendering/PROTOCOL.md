# Coordination protocol: local agents and the cloud session

Two kinds of workers touch this branch:

- **Local agents** running on the owner's workstation (the one with the RTX
  3090 and 4070 Ti Super, a full Blender build and PyTorch). They do the code,
  the builds, the renders and the training runs.
- **The cloud session** (Claude Code on claude.ai), which did the research in
  this directory. It has the full source tree but no GPU, cannot build Blender
  in a useful time, and cannot see the workstation. It is good at reading the
  codebase, writing specs and reviews, and synthesizing results.

The repository is the shared memory. The pull request for this branch is the
notification channel.

## Branch and PR

- Branch: `claude/amazing-planck-xvnke7`. All neural-rendering work lands here
  or on short-lived branches that merge into it. Do not rebase or force-push
  this branch; other sessions have it checked out.
- The draft pull request opened from this branch is watched by the cloud
  session. A comment on the PR wakes it; a push alone may not.

## How a local agent talks to the cloud session

1. Do the work, commit, push to the branch.
2. Update `STATUS.md` in the same push (what changed, what is blocked, what
   you need). Keep entries short and dated.
3. Post a PR comment. One comment per request, starting with one of:
   - `[question]` a design or codebase question the cloud session should
     answer from the tree.
   - `[review]` a request to review specific files or a commit range.
   - `[result]` a report of a build, render or training outcome, with numbers.
     Attach images by committing them under
     `doc/neural_rendering/results/<date>-<topic>/` and linking the path.
   - `[blocked]` something the cloud session can unblock by reading code or
     writing a spec.
4. The cloud session replies on the PR thread and, when the answer is code or
   a spec, pushes a commit to the branch and names it in the reply.

Keep comment bodies self-contained: file paths, commit hashes, the exact
error text in a fenced block. The cloud session may have lost its earlier
context and reads the thread cold.

## How the cloud session talks to local agents

- Specs and reviews go into this directory as files, with a PR comment
  pointing at them.
- Decisions that change the plan go into `00_synthesis.md` and `STATUS.md`.
- Requests for experiments go into `STATUS.md` under "Requested" with enough
  detail to run unattended.

## Rules for everyone

- Line numbers in the specs are from commit `afb395bd` and drift. Search for
  the symbol, do not trust the number.
- Every kernel change ships with a test or a reference render. A plausible
  image is not a passing test. Furnace test and MIS-consistency render are
  mandatory for any new closure.
- Do not put model identifiers in commit messages or code.
- Large binaries (trained weights, datasets, EXRs) do not go in the repo.
  Commit the script that regenerates them and the summary numbers.
- Nothing from NVIDIA SDKs under the RTX SDK license (RTXNTC, RTX NRC SDK,
  DLSS beyond what upstream already integrates) goes into this tree.
  tiny-cuda-nn (BSD-3) is allowed.
