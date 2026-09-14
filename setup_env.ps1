$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ProjectRoot = $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot '.venv'
$PythonExe = 'python'
$Requirements = @(
    'langchain-openai',
    'pydantic',
    'psycopg2-binary',
    'spacy',
    'tqdm',
    'streamlit',
    'sentence-transformers'
)

function Write-Diag {
    param([string]$Message)
    Write-Host ("[setup_env] {0}" -f $Message)
}

Write-Diag "Project root: $ProjectRoot"

$PythonCmd = Get-Command $PythonExe -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
    Write-Error "Python 3.11+ not found in PATH. Please install Python 3.11+ and retry."
    exit 1
}

$PyVersionText = & $PythonCmd.Source -c "import sys; print(sys.version_info.major, sys.version_info.minor)"
if (($LASTEXITCODE) -ne 0) {
    Write-Error "Failed to query Python version from $($PythonCmd.Source)."
    exit 1
}
$PyVersionParts = $PyVersionText -split '\s+'
$PyMajor = [int]$PyVersionParts[0]
$PyMinor = [int]$PyVersionParts[1]
if (($PyMajor -lt 3) -or ($PyMajor -eq 3 -and $PyMinor -lt 11)) {
    Write-Error "Python 3.11+ required; detected Python $PyMajor.$PyMinor at $($PythonCmd.Source)."
    exit 1
}
Write-Diag "Python detected: $($PythonCmd.Source) (v$PyMajor.$PyMinor)"

if (-not (Test-Path -LiteralPath (Join-Path $VenvDir 'Scripts'))) {
    Write-Diag "Creating virtual environment at $VenvDir"
    & $PythonCmd.Source -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
} else {
    Write-Diag "Virtual environment already exists at $VenvDir"
}

$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Error "Expected venv interpreter not found: $VenvPython"
    exit 1
}

Write-Diag "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip upgrade failed."
    exit 1
}

Write-Diag "Installing requirements: $($Requirements -join ', ')"
& $VenvPython -m pip install $Requirements
if ($LASTEXITCODE -ne 0) {
    Write-Error "Requirement installation failed."
    exit 1
}

Write-Diag "Downloading spaCy Chinese model: zh_core_web_sm"
& $VenvPython -m spacy download zh_core_web_sm
if ($LASTEXITCODE -ne 0) {
    Write-Error "spaCy model download failed."
    exit 1
}

Write-Diag "Verifying installed packages..."
& $VenvPython -c "import langchain_openai, pydantic, psycopg2, spacy, tqdm, streamlit, sentence_transformers; import spacy as s; nlp = s.load('zh_core_web_sm'); print('langchain_openai OK'); print('pydantic OK'); print('psycopg2 OK'); print('spacy OK'); print('tqdm OK'); print('streamlit OK'); print('sentence_transformers OK'); print('zh_core_web_sm OK: {0}'.format(nlp.meta['name']))"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Package verification failed."
    exit 1
}

Write-Diag "Environment setup complete. Activate with: .\.venv\Scripts\Activate.ps1"