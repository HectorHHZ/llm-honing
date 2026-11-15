param(
    [ValidateSet('inspect','serve','status','stop')][string]$Mode = 'serve',
    [int]$Port = 8803,
    [ValidateSet('reference','tree','tree-compiled')][string]$Inference = 'reference',
    [string]$CheckpointManifest,
    [string]$Distribution = 'Ubuntu',
    [string]$LinuxPython = '/opt/honing/venvs/nanojev/bin/python'
)
$ErrorActionPreference = 'Stop'
function Convert-GameWslPath([string]$Value) {
    if ($Value.StartsWith('/')) { return $Value }
    $resolved = (Resolve-Path -LiteralPath $Value).Path
    if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') { throw 'Expected a drive path or Linux path.' }
    return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\','/')
}
$launcher = Convert-GameWslPath (Join-Path $PSScriptRoot 'model_env.py')
$modelArgs = @('-d',$Distribution,'--exec',$LinuxPython,$launcher,$Mode,'--port',$Port,'--inference',$Inference)
if ($CheckpointManifest) { $modelArgs += @('--checkpoint-manifest',(Convert-GameWslPath $CheckpointManifest)) }
& wsl.exe @modelArgs
exit $LASTEXITCODE
