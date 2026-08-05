param(
    [string]$Python = "",
    [switch]$Recreate,
    [switch]$Cuda,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Test-UsablePython {
    param([string]$Exe)

    try {
        if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) {
            return $false
        }
    }
    catch {
        return $false
    }

$probe = @'
import platform
import sys
import sysconfig

exe = sys.executable
plat = sysconfig.get_platform()
impl = platform.python_implementation()
if impl != 'CPython':
    raise SystemExit('unsupported implementation: %s' % impl)
if 'mingw' in plat.lower() or 'msys' in exe.lower() or 'cygwin' in exe.lower():
    raise SystemExit('unsupported MSYS2/Cygwin Python: %s (%s)' % (exe, plat))
if sys.version_info < (3, 10):
    raise SystemExit('Python 3.10+ is required: %s' % sys.version)
print(exe)
print(sys.version.split()[0])
print(plat)
'@

    try {
        & $Exe -c $probe 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
        return $true
    }
    catch {
        return $false
    }
}

function Get-ProjectPython {
    if ($Python) {
        if (Test-UsablePython $Python) {
            return (Resolve-Path -LiteralPath $Python).Path
        }
        throw "The Python path passed with -Python is not usable for this project: $Python"
    }

    $candidates = @()

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $pyLauncherPath = $pyLauncher.Source
        foreach ($version in @("-3.12", "-3.11", "-3.10", "-3")) {
            try {
                $path = (& $pyLauncherPath $version -c "import sys; print(sys.executable)") 2>$null
                if ($path) {
                    $candidates += $path.Trim()
                }
            }
            catch {
            }
        }
    }

    foreach ($path in @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Program Files\Python312\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Program Files\Python310\python.exe"
    )) {
        if ($path) {
            $candidates += $path
        }
    }

    $pathPython = Get-Command python -ErrorAction SilentlyContinue
    if ($pathPython) {
        $candidates += $pathPython.Source
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-UsablePython $candidate) {
            return $candidate
        }
    }

    throw @"
Could not find a supported Windows CPython 3.10+ interpreter.

Install Python 3.12, then open a new PowerShell:
  winget install --id Python.Python.3.12 -e

Do not use MSYS2/Cygwin Python for this project. It makes pip compile NumPy
and PyTorch dependencies from source instead of installing Windows wheels.
"@
}

if ([Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw "This setup script is for Windows PowerShell only."
}

$BasePython = Get-ProjectPython
Write-Host "Using Python: $BasePython"

if ((Test-Path -LiteralPath $VenvDir) -and $Recreate) {
    Write-Host "Removing existing virtual environment: $VenvDir"
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating virtual environment: $VenvDir"
    & $BasePython -m venv $VenvDir
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Virtual environment was not created correctly. Expected: $VenvPython"
}

Write-Host "Verifying virtual environment Python..."
if (-not (Test-UsablePython $VenvPython)) {
    throw "The virtual environment Python is not usable. Re-run with -Recreate after installing Windows CPython."
}

if (-not $SkipInstall) {
    $requirements = if ($Cuda) {
        Join-Path $RepoRoot "requirements-windows-cuda.txt"
    } else {
        Join-Path $RepoRoot "requirements.txt"
    }
    Write-Host "Installing Python dependencies..."
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install --only-binary=:all: -r $requirements
}

Write-Host ""
Write-Host "Done. Activate the environment with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run a smoke test with:"
Write-Host "  python -m pytest"
Write-Host "  python -m ntn_repro.reproduce_all --fast"
