<#
.SYNOPSIS
Post a message to the coordination PR so the cloud session wakes up (Windows
counterpart of post.sh).

.DESCRIPTION
  .\post.ps1 result   "title" [body-file]     # a build/render/training outcome
  .\post.ps1 question "title" [body-file]     # a design or codebase question
  .\post.ps1 review   "title" [body-file]     # ask for a review of files or a commit range
  .\post.ps1 blocked  "title" [body-file]     # something the cloud session can unblock

The body is read from the file if given, else from the pipeline, else from -Body:
  python tools\buildbox\check_env.py | .\post.ps1 result "Build box environment"

Commit and push BEFORE posting so the cloud session can read the state you are
describing. Needs the GitHub CLI (`gh auth login`). Environment variables
NEURAL_PR_REPO (default matiasvivancob/blender) and NEURAL_PR_NUMBER (default 1)
select the PR, exactly as in post.sh. Run it from inside the repository so the
branch and commit in the header are the right ones.
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('result', 'question', 'review', 'blocked')]
    [string]$Kind,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$Title,

    [Parameter(Position = 2)]
    [string]$BodyFile,

    [string]$Body
)

# Native commands do not throw; $LASTEXITCODE is checked explicitly below.
$ErrorActionPreference = 'Continue'

$repo = if ($env:NEURAL_PR_REPO) { $env:NEURAL_PR_REPO } else { 'matiasvivancob/blender' }
$pr = if ($env:NEURAL_PR_NUMBER) { $env:NEURAL_PR_NUMBER } else { '1' }

if ($BodyFile) {
    if (-not (Test-Path -LiteralPath $BodyFile)) {
        Write-Error "body file not found: $BodyFile"
        exit 1
    }
    $text = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $BodyFile).Path)
}
elseif ($Body) {
    $text = $Body
}
else {
    # $input is the pipeline feeding this script; it is empty when nothing is piped.
    $text = ($input | Out-String)
}
if ([string]::IsNullOrWhiteSpace($text)) {
    Write-Error 'empty body: give a body file, pipe text into the script, or pass -Body'
    exit 1
}

$sha = (& git rev-parse --short HEAD 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $sha) { $sha = 'unknown' }
$branch = (& git rev-parse --abbrev-ref HEAD 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $branch) { $branch = 'unknown' }

$nl = "`n"
$message = "[$Kind] $Title" + $nl + $nl +
    'branch: `' + $branch + '` head: `' + $sha + '`' + $nl + $nl +
    $text.TrimEnd() + $nl

# gh reads the body from a UTF-8 file: piping a string to a native command uses
# $OutputEncoding, which is ASCII in Windows PowerShell 5.1 and would mangle non-ASCII text.
$tmp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), 'neural_post_' + [guid]::NewGuid().ToString('N') + '.md')
try {
    [System.IO.File]::WriteAllText($tmp, $message, (New-Object System.Text.UTF8Encoding($false)))
    & gh pr comment $pr --repo $repo --body-file $tmp
    if ($LASTEXITCODE -ne 0) {
        Write-Error ("gh pr comment failed (exit code {0}); is gh auth status green?" -f $LASTEXITCODE)
        exit 1
    }
}
finally {
    Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
}
Write-Host "posted [$Kind] $Title to $repo#$pr"
