param(
    [ValidateSet('demo', 'model', 'server', 'replay')]
    [string]$Mode = 'demo',
    [int]$Seed = 7,
    [ValidateRange(0, 4)][int]$Level = 0,
    [ValidateRange(1, 10000)][int]$MaxSteps = 100,
    [string]$DecisionServer = 'http://127.0.0.1:8803',
    [string]$RunDir,
    [string]$WslDistribution = 'Ubuntu',
    [string]$NanoJevPython,
    [string]$NanoJevSource,
    [switch]$Open
)
$ErrorActionPreference = 'Stop'
$integrationDir = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $integrationDir '..\..')).Path

if ($Mode -eq 'server') {
    Write-Host ('Starting NanoJev in ' + $WslDistribution + '. Keep this terminal open; Ctrl+C stops the service.')
    $serverArgs = @('--distribution', $WslDistribution, '--cd', $repoRoot, '--exec', '/bin/bash',
        'integrations/griddly/scripts/serve_nanojev.sh')
    if ($NanoJevPython) { $serverArgs += @('--python', $NanoJevPython) }
    if ($NanoJevSource) { $serverArgs += @('--source', $NanoJevSource) }
    & wsl.exe @serverArgs
    exit $LASTEXITCODE
}

$gamePython = Join-Path $integrationDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $gamePython)) {
    throw 'The Windows game environment is missing. See integrations/griddly/README.md.'
}
if ($Mode -eq 'replay') {
    if (-not $RunDir) { throw 'Replay mode needs -RunDir pointing to a completed game run.' }
    $completedRun = (Resolve-Path -LiteralPath $RunDir).Path
} else {
    if ($Mode -eq 'model') {
        $health = Invoke-RestMethod -Uri ($DecisionServer.TrimEnd('/') + '/v1/health') -TimeoutSec 10
        if (-not $health.ready) { throw 'The decision server is not ready.' }
        Write-Host ('Decision model: ' + $health.metadata.family)
    }
    $policyName = if ($Mode -eq 'demo') { 'baseline' } else { 'model' }
    $matchArgs = @((Join-Path $integrationDir 'scripts\match.py'), '--policy', $policyName,
        '--seed', $Seed, '--level', $Level, '--max-steps', $MaxSteps)
    if ($Mode -eq 'model') { $matchArgs += @('--decision-server', $DecisionServer) }
    if ($RunDir) { $matchArgs += @('--out', $RunDir) }
    Write-Host ('Running ' + $policyName + ' in the native Griddly engine...')
    $matchOutput = & $gamePython @matchArgs
    if ($LASTEXITCODE -ne 0) { throw 'Game execution failed. Read the error above and the run error.json.' }
    $result = ($matchOutput | Select-Object -Last 1) | ConvertFrom-Json
    $completedRun = $result.out
    Write-Host ('Result: ' + $result.status + '; steps: ' + $result.steps + '; success: ' + $result.success)
}

& $gamePython (Join-Path $integrationDir 'scripts\view_replay.py') $completedRun
if ($LASTEXITCODE -ne 0) { throw 'Replay generation failed.' }
$replayPath = Join-Path $completedRun 'replay.html'
Write-Host ('Run saved to: ' + $completedRun)
Write-Host ('Readable replay: ' + $replayPath)
if ($Open) { Invoke-Item -LiteralPath $replayPath }
