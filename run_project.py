#!/usr/bin/env python3
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
VENV_DIR = ROOT_DIR / ".venv"
IS_WINDOWS = os.name == "nt"

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_HOST = "127.0.0.1"
DEFAULT_FRONTEND_PORT = 5173

BACKEND_INSTALL = [
    "fastapi",
    "uvicorn",
    "pandas",
    "numpy",
    "joblib",
    "scikit-learn==1.7.2",
    "lifelines",
]
LOG_URL_RE = re.compile(r"http://[^\s]+")


class Colors:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"


class ProjectRunner:
    def __init__(self) -> None:
        self.processes = []
        self.shutdown_requested = False
        self.frontend_port = DEFAULT_FRONTEND_PORT
        self.backend_url = f"http://{BACKEND_HOST}:{BACKEND_PORT}/"
        self.frontend_url = None

    def log(self, prefix: str, message: str, color: str = Colors.CYAN) -> None:
        print(f"{color}{prefix}{Colors.RESET} {message}", flush=True)

    def run_checked(self, cmd, cwd: Path, env=None, label="CMD") -> None:
        self.log(f"[{label}]", " ".join(cmd), Colors.BLUE)
        result = subprocess.run(cmd, cwd=cwd, env=env, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{label} failed with exit code {result.returncode}")

    def ensure_command(self, command: str) -> None:
        if shutil.which(command):
            return
        raise RuntimeError(f"Required command not found: {command}")

    def ensure_venv(self) -> Path:
        if not VENV_DIR.exists():
            self.log("[SETUP]", "Creating Python virtual environment", Colors.GREEN)
            self.run_checked([sys.executable, "-m", "venv", str(VENV_DIR)], ROOT_DIR, label="VENV")
        return VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")

    def pip_path(self) -> Path:
        return VENV_DIR / ("Scripts/pip.exe" if IS_WINDOWS else "bin/pip")

    def backend_imports_ready(self, python_path: Path) -> bool:
        cmd = [
            str(python_path),
            "-c",
            (
                "import fastapi, uvicorn, pandas, numpy, joblib, sklearn, lifelines; "
                "import sys; "
                "sys.exit(0 if sklearn.__version__.startswith('1.7.2') else 1)"
            ),
        ]
        result = subprocess.run(cmd, cwd=ROOT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0

    def ensure_backend_dependencies(self) -> Path:
        python_path = self.ensure_venv()
        if self.backend_imports_ready(python_path):
            self.log("[SETUP]", "Backend dependencies already available", Colors.GREEN)
            return python_path

        self.log("[SETUP]", "Installing backend dependencies", Colors.GREEN)
        self.run_checked([str(self.pip_path()), "install", *BACKEND_INSTALL], ROOT_DIR, label="PIP")
        return python_path

    def ensure_frontend_dependencies(self) -> None:
        self.ensure_command("npm")
        node_modules = FRONTEND_DIR / "node_modules"
        if node_modules.exists():
            self.log("[SETUP]", "Frontend dependencies already available", Colors.GREEN)
            return

        self.log("[SETUP]", "Installing frontend dependencies", Colors.GREEN)
        self.run_checked(["npm", "install"], FRONTEND_DIR, label="NPM")

    def is_port_open(self, host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0

    def wait_for_port(self, host: str, port: int, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_port_open(host, port):
                return True
            time.sleep(0.25)
        return False

    def wait_for_http(self, url: str, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status < 500:
                        return True
            except (urllib.error.URLError, TimeoutError, ValueError):
                time.sleep(0.5)
        return False

    def is_project_process(self, proc) -> bool:
        try:
            cmdline = " ".join(proc.cmdline())
            cwd = proc.cwd() if hasattr(proc, "cwd") else ""
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False

        markers = [str(ROOT_DIR), str(BACKEND_DIR), str(FRONTEND_DIR), "uvicorn", "vite", "npm run dev"]
        haystack = f"{cmdline}\n{cwd}"
        return any(marker in haystack for marker in markers)

    def process_on_port(self, port: int):
        if psutil is None:
            return None

        try:
            for conn in psutil.net_connections(kind="inet"):
                if conn.laddr and conn.laddr.port == port and conn.pid:
                    try:
                        return psutil.Process(conn.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        return None
        except psutil.Error:
            return None
        return None

    def terminate_process(self, proc) -> None:
        if psutil is None:
            return
        try:
            children = proc.children(recursive=True)
            for child in children:
                child.terminate()
            proc.terminate()
            gone, alive = psutil.wait_procs(children + [proc], timeout=4)
            for item in alive:
                item.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return

    def ensure_backend_port(self) -> None:
        proc = self.process_on_port(BACKEND_PORT)
        if proc is None:
            return

        if self.is_project_process(proc):
            self.log("[CLEANUP]", f"Stopping existing project backend on port {BACKEND_PORT}", Colors.YELLOW)
            self.terminate_process(proc)
            time.sleep(1)
            return

        raise RuntimeError(f"Port {BACKEND_PORT} is already in use by PID {proc.pid}. Stop it and rerun.")

    def pick_frontend_port(self) -> int:
        port = DEFAULT_FRONTEND_PORT
        while port < DEFAULT_FRONTEND_PORT + 20:
            proc = self.process_on_port(port)
            if proc is None:
                return port
            if self.is_project_process(proc):
                self.log("[CLEANUP]", f"Stopping existing project frontend on port {port}", Colors.YELLOW)
                self.terminate_process(proc)
                time.sleep(1)
                if not self.is_port_open(FRONTEND_HOST, port):
                    return port
            else:
                port += 1
        raise RuntimeError("No free frontend port found in range 5173-5192")

    def sanitize_line(self, line: str) -> str:
        line = LOG_URL_RE.sub("[hidden-url]", line).strip()
        if not line:
            return ""
        ignore_fragments = [
            "Local:",
            "Network:",
            "Uvicorn running on",
            "Application startup complete.",
            "Waiting for application startup.",
        ]
        if any(fragment in line for fragment in ignore_fragments):
            return ""
        return line

    def stream_output(self, process: subprocess.Popen, prefix: str, color: str) -> None:
        try:
            for raw_line in iter(process.stdout.readline, ""):
                if not raw_line:
                    break
                line = self.sanitize_line(raw_line)
                if line:
                    self.log(prefix, line, color)
        except Exception:
            return

    def start_backend(self, python_path: Path) -> subprocess.Popen:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [
            str(python_path),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            BACKEND_HOST,
            "--port",
            str(BACKEND_PORT),
        ]
        self.log("[START]", "Starting backend", Colors.GREEN)
        process = subprocess.Popen(
            cmd,
            cwd=BACKEND_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.processes.append(process)
        threading.Thread(
            target=self.stream_output,
            args=(process, "[BACKEND]", Colors.BLUE),
            daemon=True,
        ).start()
        return process

    def start_frontend(self, port: int) -> subprocess.Popen:
        cmd = ["npm", "run", "dev", "--", "--host", FRONTEND_HOST, "--port", str(port)]
        self.log("[START]", "Starting frontend", Colors.GREEN)
        process = subprocess.Popen(
            cmd,
            cwd=FRONTEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.processes.append(process)
        threading.Thread(
            target=self.stream_output,
            args=(process, "[FRONTEND]", Colors.CYAN),
            daemon=True,
        ).start()
        return process

    def ensure_running(self, process: subprocess.Popen, name: str) -> None:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited unexpectedly with code {process.returncode}")

    def install_and_start(self) -> None:
        self.ensure_command("python3" if shutil.which("python3") else "python")
        self.ensure_command("npm")

        python_path = self.ensure_backend_dependencies()
        self.ensure_frontend_dependencies()

        self.ensure_backend_port()
        self.frontend_port = self.pick_frontend_port()

        backend_proc = self.start_backend(python_path)
        if not self.wait_for_http(self.backend_url, timeout=60):
            self.ensure_running(backend_proc, "Backend")
            raise RuntimeError("Backend did not become ready within 60 seconds")

        frontend_proc = self.start_frontend(self.frontend_port)
        frontend_url = f"http://localhost:{self.frontend_port}"
        self.frontend_url = frontend_url
        if not self.wait_for_http(f"http://{FRONTEND_HOST}:{self.frontend_port}/", timeout=60):
            self.ensure_running(frontend_proc, "Frontend")
            raise RuntimeError("Frontend did not become ready within 60 seconds")

        print(frontend_url, flush=True)

        while not self.shutdown_requested:
            self.ensure_running(backend_proc, "Backend")
            self.ensure_running(frontend_proc, "Frontend")
            time.sleep(1)

    def shutdown(self, *_args) -> None:
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    continue
        deadline = time.time() + 5
        while time.time() < deadline:
            alive = [proc for proc in self.processes if proc.poll() is None]
            if not alive:
                return
            time.sleep(0.25)
        for process in self.processes:
            if process.poll() is None:
                try:
                    process.kill()
                except Exception:
                    continue
def main() -> int:
    runner = ProjectRunner()
    signal.signal(signal.SIGINT, runner.shutdown)
    signal.signal(signal.SIGTERM, runner.shutdown)
    try:
        runner.install_and_start()
        return 0
    except KeyboardInterrupt:
        runner.shutdown()
        return 0
    except Exception as exc:
        runner.log("[ERROR]", str(exc), Colors.RED)
        runner.shutdown()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
