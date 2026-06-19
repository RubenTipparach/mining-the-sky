@echo off
REM Build and run the desktop (native) client in release mode.
setlocal
cd /d "%~dp0"

REM Keep the Rust toolchain current; some crates (egui) need a recent rustc.
where rustup >nul 2>nul && (
    echo Updating the Rust toolchain...
    rustup update --no-self-update stable
)

echo Building and running the desktop client...
cargo run --release -p app %*
if errorlevel 1 (
    echo.
    echo Build or run failed. See the output above.
    exit /b 1
)

endlocal
