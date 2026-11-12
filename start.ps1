# Start API (port 8001) + Streamlit UI in one step.
# API and chat_ui both use APP_PORT from settings (default 8001).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Free the ports if a previous run is still hanging
foreach ($port in 8001, 8501) {
    $pids = (Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue).OwningProcess | Select-Object -Unique
    foreach ($pid in $pids) {
        if ($pid -and $pid -ne 0) {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }
}

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$PSScriptRoot'; .\venv\Scripts\python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001"
)

.\venv\Scripts\streamlit run chat_ui.py
