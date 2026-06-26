param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,
    [Parameter(Mandatory = $true)]
    [string]$RemoteUser,
    [Parameter(Mandatory = $true)]
    [string]$StablePath,
    [Parameter(Mandatory = $true)]
    [string]$DevBasePath
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$archiveDir = Join-Path $repoRoot ".tmp"
$archivePath = Join-Path $archiveDir "ai-dev-code-$timestamp.tar.gz"
$remoteArchive = "/tmp/ai-dev-code-$timestamp.tar.gz"
$releasePath = "$DevBasePath/releases/$timestamp"

$sourcePaths = @(
    "ai",
    "scripts",
    "stream",
    "detector",
    "tracking",
    "messaging",
    "rules",
    "configs",
    "tools",
    "tests",
    "config.py",
    "main.py",
    "serve_mjpeg.py",
    "requirements.txt"
)

$missing = $sourcePaths | Where-Object { -not (Test-Path (Join-Path $repoRoot $_)) }
if ($missing) {
    throw "Missing deployment paths: $($missing -join ', ')"
}

New-Item -ItemType Directory -Force -Path $archiveDir | Out-Null

Write-Host "[1/3] Packaging local source code..."
& tar.exe -czf $archivePath -C $repoRoot @sourcePaths
if ($LASTEXITCODE -ne 0) {
    throw "tar failed with exit code $LASTEXITCODE"
}

Write-Host "[2/3] Uploading one archive to GPU PC..."
& scp.exe $archivePath "${RemoteUser}@${RemoteHost}:$remoteArchive"
if ($LASTEXITCODE -ne 0) {
    throw "scp failed with exit code $LASTEXITCODE"
}

$remoteCommand = @"
set -eu
mkdir -p '$releasePath'
tar -xzf '$remoteArchive' -C '$releasePath'
rm -f '$remoteArchive'
find '$releasePath' -name '*.sh' -exec sed -i 's/\r$//' {} +
ln -sfn '$StablePath/.venv' '$releasePath/.venv'
if [ -f '$StablePath/yolo26n-pose.pt' ]; then
  ln -sfn '$StablePath/yolo26n-pose.pt' '$releasePath/yolo26n-pose.pt'
fi
mkdir -p '$releasePath/benchmark'
if [ -d '$StablePath/benchmark/results' ]; then
  ln -sfn '$StablePath/benchmark/results' '$releasePath/benchmark/results'
fi
ln -sfn '$releasePath' '$DevBasePath/current'
printf 'GPU dev release ready: %s\n' '$releasePath'
"@

Write-Host "[3/3] Creating an isolated release and switching current..."
$remoteCommand = $remoteCommand -replace "`r`n", "`n"
& ssh.exe "${RemoteUser}@${RemoteHost}" $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "remote setup failed with exit code $LASTEXITCODE"
}

Remove-Item -LiteralPath $archivePath -Force

Write-Host ""
Write-Host "Deployment complete."
Write-Host "GPU current path: $DevBasePath/current"
Write-Host "Stable Git path was not modified: $StablePath"
