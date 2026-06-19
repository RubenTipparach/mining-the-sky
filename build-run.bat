@echo off
REM Build and run the desktop (native) client.
REM   build-run.bat        optimized release build (best runtime perf)
REM   build-run.bat dev    fast debug build (quicker compile, slower runtime)
REM Note: a clean release build compiles wgpu + egui with thin LTO and can take
REM a few minutes - it is not stuck, it just goes quiet during the link step.
setlocal
cd /d "%~dp0"

if /I "%~1"=="dev" (
    echo Building and running the desktop client ^(debug - fast compile^)...
    cargo run -p app
) else (
    echo Building and running the desktop client ^(release^)...
    echo A clean release build can take a few minutes; subsequent runs are fast.
    cargo run --release -p app %*
)
if errorlevel 1 (
    echo.
    echo Build or run failed. See the output above.
    echo If a crate reports it needs a newer rustc, run: rustup update
    exit /b 1
)

endlocal
