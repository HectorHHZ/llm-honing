param(
    [ValidateSet('play','demo','model','replay','verify','test')][string]$Mode = 'play',
    [string]$Layout = 'cramped_room',
    [int]$MaxSteps = 400,
    [int]$Port = 4185,
    [string]$DecisionServer,
    [string]$RunDir,
    [switch]$OldDynamics,
    [switch]$Open
)
$ErrorActionPreference = 'Stop'
$gamePython = Join-Path $PSScriptRoot '.venv/python.exe'
if (-not (Test-Path -LiteralPath $gamePython)) { $gamePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $gamePython)) { throw 'Run integrations/overcooked/setup.ps1 first.' }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Push-Location -LiteralPath $repoRoot
try {
    if ($Mode -eq 'test') {
        $testTemp = Join-Path $PSScriptRoot ('.cache/pytest-' + [guid]::NewGuid().ToString('N'))
        & $gamePython -m pytest (Join-Path $PSScriptRoot 'tests') -q --basetemp $testTemp -o ('cache_dir=' + (Join-Path $PSScriptRoot '.cache/pytest-cache'))
    } elseif ($Mode -eq 'verify') {
        if (-not $RunDir) { throw '-RunDir is required.' }
        & $gamePython -m integrations.overcooked.verify $RunDir
    } elseif ($Mode -in @('demo','model')) {
        $matchArgs = @('-m','integrations.overcooked.match','--policy',$(if ($Mode -eq 'model') {'model'} else {'baseline'}),
                       '--layout',$Layout,'--max-steps',$MaxSteps,'--verify')
        if ($Mode -eq 'model' -and -not $DecisionServer) { throw 'Model mode requires -DecisionServer.' }
        if ($DecisionServer) { $matchArgs += @('--decision-server',$DecisionServer) }
        if ($RunDir) { $matchArgs += @('--out',$RunDir) }
        if ($OldDynamics) { $matchArgs += '--old-dynamics' }
        & $gamePython @matchArgs
    } else {
        $uiArgs = @('-m','integrations.overcooked.ui','--port',$Port,'--layout',$Layout,'--max-steps',$MaxSteps)
        if ($DecisionServer) { $uiArgs += @('--decision-server',$DecisionServer) }
        if ($OldDynamics) { $uiArgs += '--old-dynamics' }
        if ($Mode -eq 'replay') {
            if (-not $RunDir) { throw 'Replay mode requires -RunDir.' }
            $uiArgs += @('--replay',$RunDir)
        }
        if ($Open) { $uiArgs += '--open' }
        & $gamePython @uiArgs
    }
    $gameExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $gameExitCode
