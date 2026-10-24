<#
.SYNOPSIS
Poll the coordination PR for new comments and new commits on the branch
(Windows counterpart of watch.sh).

.DESCRIPTION
Prints every PR comment not seen before and fast-forwards the current branch
when origin has new commits (git pull --ff-only; never rebases).

  .\watch.ps1                # one pass
  .\watch.ps1 -Loop 300      # every 300 seconds (5 minutes), until Ctrl+C

Environment: NEURAL_PR_REPO (default matiasvivancob/blender), NEURAL_PR_NUMBER
(default 1), NEURAL_WATCH_STATE (default %LOCALAPPDATA%\neural_pr_watch, where
the file seen_comment_ids remembers which comments were printed). Run it from
inside the repository checkout of the coordination branch.
#>
param(
    [int]$Loop = 0
)

# Native commands do not throw (and in Windows PowerShell 5.1 stderr text would
# become a terminating error under 'Stop'); $LASTEXITCODE is checked explicitly.
$ErrorActionPreference = 'Continue'

$repo = if ($env:NEURAL_PR_REPO) { $env:NEURAL_PR_REPO } else { 'matiasvivancob/blender' }
$pr = if ($env:NEURAL_PR_NUMBER) { $env:NEURAL_PR_NUMBER } else { '1' }
if ($env:NEURAL_WATCH_STATE) {
    $stateDir = $env:NEURAL_WATCH_STATE
} else {
    $stateDir = Join-Path $env:LOCALAPPDATA 'neural_pr_watch'
}
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$seenFile = Join-Path $stateDir 'seen_comment_ids'
if (-not (Test-Path -LiteralPath $seenFile)) {
    New-Item -ItemType File -Path $seenFile | Out-Null
}

function Invoke-WatchPass {
    $seen = New-Object 'System.Collections.Generic.HashSet[string]'
    foreach ($line in [System.IO.File]::ReadAllLines($seenFile)) {
        if ($line.Trim()) { [void]$seen.Add($line.Trim()) }
    }

    # One JSON object per line (tojson escapes newlines inside the body, so a
    # multi-line comment stays on one line).
    $jq = '.[] | {id: .id, user: .user.login, created_at: .created_at, body: .body} | tojson'
    $lines = & gh api "repos/$repo/issues/$pr/comments?per_page=100" --paginate --jq $jq
    if ($LASTEXITCODE -ne 0) {
        Write-Warning ("gh api failed (exit code {0}); is gh auth status green?" -f $LASTEXITCODE)
    }
    else {
        foreach ($line in @($lines)) {
            if (-not "$line".Trim()) { continue }
            $c = $line | ConvertFrom-Json
            $id = [string]$c.id
            if (-not $seen.Contains($id)) {
                Write-Host ''
                Write-Host "=== PR comment $id by $($c.user) at $($c.created_at) ==="
                Write-Host $c.body
                Add-Content -LiteralPath $seenFile -Value $id
                [void]$seen.Add($id)
            }
        }
    }

    & git fetch -q origin
    $branch = (& git rev-parse --abbrev-ref HEAD 2>$null)
    $behind = '0'
    if ($LASTEXITCODE -eq 0 -and $branch) {
        $n = (& git rev-list --count "HEAD..origin/$branch" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $n) { $behind = "$n".Trim() }
    }
    if ($behind -ne '0') {
        Write-Host ''
        Write-Host "=== $behind new commit(s) on origin, pulling (ff-only) ==="
        & git log --oneline "HEAD..origin/$branch"
        & git pull -q --ff-only
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'ff-only pull failed: local commits diverged, merge by hand'
        }
    }
}

if ($Loop -gt 0) {
    while ($true) {
        Invoke-WatchPass
        Start-Sleep -Seconds $Loop
    }
}
else {
    Invoke-WatchPass
}
