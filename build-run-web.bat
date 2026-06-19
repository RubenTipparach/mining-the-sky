@echo off
REM Build and serve the web (WebGPU / wasm) client with Trunk.
REM Open the printed URL (default http://127.0.0.1:8080) in a WebGPU-capable
REM browser, e.g. recent Chrome or Edge.
setlocal
cd /d "%~dp0"

REM Keep the Rust toolchain current; some crates (egui) need a recent rustc.
where rustup >nul 2>nul && (
    echo Updating the Rust toolchain...
    rustup update --no-self-update stable
)

REM Trunk drives the wasm build; install once if missing.
where trunk >nul 2>nul
if errorlevel 1 (
    echo Trunk is not installed. Installing it now...
    cargo install --locked trunk
    if errorlevel 1 (
        echo Failed to install Trunk. Run "cargo install --locked trunk" manually.
        exit /b 1
    )
)

REM Make sure the wasm target is available.
rustup target add wasm32-unknown-unknown >nul 2>nul

echo Serving the web client. Press Ctrl+C to stop.
cd crates\app
trunk serve %*
if errorlevel 1 (
    echo.
    echo Build or serve failed. See the output above.
    exit /b 1
)

endlocal
