$ErrorActionPreference = "Stop"

# This script lives in scripts\jaad; the project root is two folders up.
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$JaadRoot = Join-Path $ProjectRoot "data\JAAD"
$ArchivePath = Join-Path $JaadRoot "JAAD_clips.zip"
$ClipsPath = Join-Path $JaadRoot "JAAD_clips"
$RepositoryUrl = "https://github.com/ykotseruba/JAAD.git"
$RepositoryBranch = "JAAD_2.0"
$OfficialClipsUrl = "http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip"
$GoogleDriveId = "1HCFLBO9fJutCKG11FtjKfdLvME6Qe_5L"

if (-not (Test-Path $JaadRoot)) {
    Write-Host "Downloading the official JAAD annotations..."
    git clone --branch $RepositoryBranch --depth 1 $RepositoryUrl $JaadRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Could not clone the official JAAD repository."
    }
}
elseif (-not (Test-Path (Join-Path $JaadRoot "annotations"))) {
    throw "The path already exists but is not a complete JAAD repository: $JaadRoot"
}
else {
    Write-Host "JAAD annotations already exist: $JaadRoot"
}

$ExistingClips = @()
if (Test-Path $ClipsPath) {
    $ExistingClips = @(Get-ChildItem $ClipsPath -Filter "video_*.mp4" -File)
}

if ($ExistingClips.Count -lt 346) {
    Write-Host "Downloading approximately 3.1 GB of JAAD video clips..."
    curl.exe --location --fail --retry 5 --continue-at - --output $ArchivePath $OfficialClipsUrl

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "The York University download failed. Trying the official Google Drive mirror."
        uvx --from gdown gdown "https://drive.google.com/uc?id=$GoogleDriveId" --output $ArchivePath
        if ($LASTEXITCODE -ne 0) {
            throw "Both official JAAD video download sources failed."
        }
    }

    Write-Host "Extracting JAAD clips..."
    Expand-Archive -Path $ArchivePath -DestinationPath $JaadRoot -Force
}

$DownloadedClips = @(Get-ChildItem $ClipsPath -Filter "video_*.mp4" -File)
if ($DownloadedClips.Count -ne 346) {
    throw "Expected 346 JAAD clips but found $($DownloadedClips.Count) in $ClipsPath"
}

Write-Host "JAAD download is complete."
Write-Host "Dataset: $JaadRoot"
Write-Host "Video clips: $($DownloadedClips.Count)"
