param([string]$Conda = 'D:\anaconda3\Scripts\conda.exe')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$pins = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'pins.json') -Raw | ConvertFrom-Json
$cacheDir = Join-Path $PSScriptRoot '.cache'
$envDir = Join-Path $PSScriptRoot '.venv'
$upstreamDir = Join-Path $PSScriptRoot 'upstream'
$gamePython = Join-Path $envDir 'python.exe'
New-Item -ItemType Directory -Force -Path (Join-Path $cacheDir 'tmp') | Out-Null
$originalTmp, $originalTemp = $env:TMP, $env:TEMP
$originalPkgs, $originalEnvs = $env:CONDA_PKGS_DIRS, $env:CONDA_ENVS_PATH
$env:TMP = Join-Path $cacheDir 'tmp'
$env:TEMP = $env:TMP
$env:CONDA_PKGS_DIRS = Join-Path $cacheDir 'conda-pkgs'
$env:CONDA_ENVS_PATH = $PSScriptRoot
Push-Location -LiteralPath $repoRoot
function Invoke-GameSetupCommand {
    param([string]$Executable, [string[]]$NativeArgs)
    $savedPreference = $ErrorActionPreference
    try {
        # Windows PowerShell turns redirected native stderr into ErrorRecords,
        # including harmless pip warnings. Decide success using the exit code.
        $ErrorActionPreference = 'Continue'
        & $Executable @NativeArgs 2>&1 | ForEach-Object { Write-Host $_.ToString() }
        $nativeExitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedPreference }
    if ($nativeExitCode -ne 0) { throw ('Setup command failed with exit code ' + $nativeExitCode) }
}
try {
    if (-not (Test-Path -LiteralPath $gamePython)) {
        if (-not (Test-Path -LiteralPath $Conda)) { throw 'Pass -Conda with the path to your conda executable.' }
        Invoke-GameSetupCommand $Conda @('create','--prefix',$envDir,'python=3.10','pip','--no-default-packages','--yes','--quiet')
    }
    Invoke-GameSetupCommand $gamePython @('-c','import sys; assert sys.version_info[:2] == (3,10)')
    if (-not (Test-Path -LiteralPath (Join-Path $upstreamDir '.git'))) {
        Invoke-GameSetupCommand 'git' @('clone','--depth','1','--filter=blob:none','--sparse',$pins.repository,$upstreamDir)
    }
    Invoke-GameSetupCommand 'git' @('-C',$upstreamDir,'sparse-checkout','set','src/overcooked_ai_py')
    $revision = (& git -C $upstreamDir rev-parse HEAD).Trim()
    if ($revision -ne $pins.revision) {
        Invoke-GameSetupCommand 'git' @('-C',$upstreamDir,'fetch','--depth','1','origin',$pins.revision)
        Invoke-GameSetupCommand 'git' @('-C',$upstreamDir,'checkout','--detach',$pins.revision)
    }
    Invoke-GameSetupCommand $gamePython @('-m','pip','install','--disable-pip-version-check','--no-warn-script-location',
        '--cache-dir',(Join-Path $cacheDir 'pip'),'-r',(Join-Path $PSScriptRoot 'requirements-lock.txt'))
    Invoke-GameSetupCommand $gamePython @('-m','pip','install','--disable-pip-version-check','--no-warn-script-location',
        '--no-deps','--no-build-isolation','--cache-dir',(Join-Path $cacheDir 'pip'),'-e',$upstreamDir)
    Invoke-GameSetupCommand $gamePython @('-c','from integrations.overcooked.engine import runtime_identity; print(runtime_identity())')
} finally {
    Pop-Location
    $env:TMP, $env:TEMP = $originalTmp, $originalTemp
    $env:CONDA_PKGS_DIRS, $env:CONDA_ENVS_PATH = $originalPkgs, $originalEnvs
}
Write-Host 'Ready: .\integrations\overcooked\run.ps1 -Mode play -Open'
