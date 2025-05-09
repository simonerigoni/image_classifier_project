# Clean up
# .\my_scripts\Clean-Up.ps1

$currentScriptFolder = $PSScriptRoot
$currentScriptFolderLeaf = Split-Path -Path $currentScriptFolder -Leaf
. ".\$currentScriptFolderLeaf\Config.ps1"

# Stop on any error for robust execution
$ErrorActionPreference = "Stop"

Write-Host "Starting cleanup process..." -ForegroundColor Green

# Check and delete virtual environment
Write-Host "Checking for virtual environment ($venvFolder)..." -ForegroundColor Yellow
if (Test-Path $venvFolder) {
    Write-Host "Deleting virtual environment..." -ForegroundColor Yellow
    try {
        Remove-Item -Path $venvFolder -Recurse -Force
        Write-Host "Virtual environment deleted!" -ForegroundColor Green
    } catch {
        Write-Host "Failed to delete virtual environment!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Virtual environment does not exist!" -ForegroundColor Green
}

# Delete specific files if needed
$dataFolder = ".\data"
$filesToDelete = @("checkpoint.pth")

Write-Host "Checking for specific files in $dataFolder folder..." -ForegroundColor Yellow
if (Test-Path $dataFolder) {
    foreach ($file in $filesToDelete) {
        $filePath = Join-Path -Path $dataFolder -ChildPath $file
        if (Test-Path $filePath) {
            Write-Host "Deleting $file..." -ForegroundColor Yellow
            try {
                Remove-Item -Path $filePath -Force
                Write-Host "$file deleted!" -ForegroundColor Green
            } catch {
                Write-Host "Failed to delete $file!" -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "$file does not exist!" -ForegroundColor Green
        }
    }
} else {
    Write-Host "$dataFolder folder does not exist!" -ForegroundColor Green
}

Write-Host "Cleanup complete!" -ForegroundColor Green