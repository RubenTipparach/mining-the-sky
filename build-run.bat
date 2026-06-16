@echo off
REM Build and run the desktop (native) client in release mode.
setlocal
cd /d "%~dp0"

echo Building and running the desktop client...
cargo run --release -p app %*
if errorlevel 1 (
    echo.
    echo Build or run failed. See the output above.
    exit /b 1
)

endlocal
