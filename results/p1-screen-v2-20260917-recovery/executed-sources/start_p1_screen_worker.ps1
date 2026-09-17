param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('gpu', 'cpu')]
    [string]$Worker,
    [switch]$CheckOnly
)

# Keep a real console Python executable with redirected streams. The supervisor
# retains the same frozen protocol, locks, memory gates and fail-closed policy.
$ErrorActionPreference = 'Stop'
$a2Python = (Resolve-Path -LiteralPath 'F:\conda-envs\yolo-master\python.exe').Path
$a2Script = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'supervise_p1_screen.py')).Path
$a2Root = Split-Path -Parent $PSScriptRoot
if ($CheckOnly) {
    Push-Location $PSScriptRoot
    try {
        & $a2Python -X utf8 -c 'from p1_screen_contract import verify_frozen; verify_frozen(); print(True)'
        exit $LASTEXITCODE
    } finally { Pop-Location }
}
$a2LogRoot = 'F:\YOLO-Master-A2-P1\p1-screen-v2-20260915\logs'
if (-not (Test-Path -LiteralPath $a2LogRoot -PathType Container)) { throw 'Campaign log directory missing' }
$a2Stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ') + '-' + [Guid]::NewGuid().ToString('N')
$a2Out = Join-Path $a2LogRoot ('supervisor-{0}-{1}.stdout.log' -f $Worker, $a2Stamp)
$a2Err = Join-Path $a2LogRoot ('supervisor-{0}-{1}.stderr.log' -f $Worker, $a2Stamp)
if ((Test-Path -LiteralPath $a2Out) -or (Test-Path -LiteralPath $a2Err)) { throw 'Log path already exists' }
$a2Args = '-X utf8 "{0}" --worker {1}' -f $a2Script, $Worker
$a2Process = Start-Process -FilePath $a2Python -ArgumentList $a2Args -WorkingDirectory $a2Root -WindowStyle Hidden -RedirectStandardOutput $a2Out -RedirectStandardError $a2Err -Wait -PassThru
exit $a2Process.ExitCode
