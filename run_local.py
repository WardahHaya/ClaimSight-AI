import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def run_command(command: list[str], workdir: Path) -> None:
    completed = subprocess.run(command, cwd=workdir, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frontend and run the integrated ClaimSight localhost app.")
    parser.add_argument("--skip-install", action="store_true", help="Skip npm install before building the frontend.")
    parser.add_argument("--skip-build", action="store_true", help="Skip npm build and serve the existing frontend dist.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    args = parser.parse_args()

    npm_command = "npm.cmd" if os.name == "nt" else "npm"

    if not args.skip_build:
        if not args.skip_install:
            run_command([npm_command, "install"], FRONTEND_DIR)
        run_command([npm_command, "run", "build"], FRONTEND_DIR)

    uvicorn_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    raise SystemExit(uvicorn_process.returncode)


if __name__ == "__main__":
    main()
