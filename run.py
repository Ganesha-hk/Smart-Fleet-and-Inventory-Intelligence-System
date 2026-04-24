#!/usr/bin/env python3
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"


def run_command(command: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.Popen:
    return subprocess.Popen(command, shell=True, cwd=cwd, env=env)


def terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def ensure_dependencies() -> None:
    if shutil.which("npm") is None:
        print("npm not found. Install Node.js.")
        sys.exit(1)

    if shutil.which("python") is None and shutil.which("python3") is None:
        print("Python not found.")
        sys.exit(1)


def main() -> None:
    print("Starting Smart Fleet Intelligence System...\n")

    ensure_dependencies()

    is_windows = platform.system() == "Windows"
    python_cmd = "python" if is_windows else "python3"
    backend_cmd = f"{python_cmd} -m uvicorn app.main:app --reload"
    frontend_cmd = "npm run dev"

    backend = None
    frontend = None

    try:
        print("Starting Backend...")
        backend_env = os.environ.copy()
        backend_env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + backend_env.get("PYTHONPATH", "")
        backend = run_command(backend_cmd, cwd=BACKEND_DIR, env=backend_env)

        print("Starting Frontend...")
        frontend = run_command(frontend_cmd, cwd=FRONTEND_DIR)

        print("\nSystem running:")
        print("  Backend:  http://localhost:8000")
        print("  Frontend: http://localhost:5173\n")
        print("Press CTRL+C to stop...\n")

        backend.wait()
        frontend.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")
        terminate_process(backend)
        terminate_process(frontend)
        print("System stopped cleanly.")

    except Exception as exc:
        print(f"Error: {exc}")
        terminate_process(backend)
        terminate_process(frontend)
        sys.exit(1)


if __name__ == "__main__":
    main()
