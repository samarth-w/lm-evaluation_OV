<#
Simple PowerShell wrapper to run OVMS (ovms.exe) built natively on Windows.

Usage:
  ./scripts/run_ovms_native.ps1 -ModelConfigPath C:\path\to\models\config.json -OvmsExePath C:\path\to\ovms.exe

#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ModelConfigPath,

    [Parameter(Mandatory=$false)]
    [string]$OvmsExePath = "./build/bin/Release/ovms.exe",

    [int]$RestPort = 8000
)

if (-not (Test-Path $ModelConfigPath)) {
    Write-Error "Model config path not found: $ModelConfigPath"
    exit 2
}

if (-not (Test-Path $OvmsExePath)) {
    Write-Error "OVMS executable not found: $OvmsExePath"
    exit 3
}

Write-Host "Starting OVMS: $OvmsExePath --rest_port $RestPort --config_path $ModelConfigPath"

Start-Process -FilePath $OvmsExePath -ArgumentList @("--rest_port", "$RestPort", "--config_path", "$ModelConfigPath") -NoNewWindow -Wait
