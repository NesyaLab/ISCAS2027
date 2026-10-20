"""Create the local environment, then launch Jupyter, check, run, or package the project."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
import zipfile


ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".venv"
PYTHON = ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
NOTEBOOK = "iscas2027_experiments.ipynb"


def package_project(root):
    """Exclude machine-specific environments and caches; retain code and results."""
    archive = root.parent / (root.name + "_share.zip")
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as out:
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not (
                d.startswith(".venv") or d in {
                    "__pycache__", ".runtime", ".ipynb_checkpoints", ".git",
                } or (Path(directory) / d).is_symlink()
            )]
            for name in files:
                path = Path(directory) / name
                if path.is_symlink() or path.suffix == ".pyc":
                    continue
                out.write(path, Path(root.name) / path.relative_to(root))
    temporary.replace(archive)
    return archive


def prepare_environment():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12: py -3.12 start.py (Windows), or python3.12 start.py.")
    if ENV.is_symlink() or ENV.resolve().parent != ROOT:
        raise RuntimeError("Refusing to modify an environment outside this project.")
    requirements = ROOT / "requirements-lock.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    stamp = ENV / "project-environment.json"
    recorded = json.loads(stamp.read_text()) if stamp.exists() else {}
    expected = {"project": str(ROOT), "requirements_sha256": digest}
    if recorded.get("project") not in (None, str(ROOT)):
        # Preserve a copied/moved environment rather than deleting recipient files.
        backup = ROOT / (".venv-moved-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
        if backup.resolve().parent != ROOT:
            raise RuntimeError("Invalid environment backup path.")
        ENV.rename(backup)
        recorded = {}
        print(f"Preserved moved environment as {backup.name}.", flush=True)
    if not PYTHON.exists():
        print("Creating .venv with Python 3.12...", flush=True)
        venv.EnvBuilder(with_pip=True).create(ENV)
    if recorded != expected:
        subprocess.run(
            [str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
             "-r", str(requirements)], check=True, cwd=ROOT,
        )
        stamp.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    subprocess.run(
        [str(PYTHON), "-m", "ipykernel", "install", "--prefix", str(ENV),
         "--name", "iscas2027", "--display-name", "ISCAS 2027 (.venv)"],
        check=True, cwd=ROOT,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Set up and run the tests.")
    group.add_argument("--run", action="store_true", help="Set up and execute the whole notebook.")
    group.add_argument("--package", action="store_true", help="Build a shareable ZIP beside this folder.")
    args = parser.parse_args()
    if args.package:
        print(package_project(ROOT))
        return
    prepare_environment()
    if args.check:
        subprocess.run([str(PYTHON), "-m", "pip", "check"], check=True, cwd=ROOT)
        subprocess.run([str(PYTHON), "-m", "unittest", "-v"], check=True, cwd=ROOT)
    elif args.run:
        script = """
from pathlib import Path
import nbformat
from nbclient import NotebookClient
p = Path("iscas2027_experiments.ipynb")
nb = nbformat.read(p, as_version=4)
def progress(cell, cell_index):
    if cell.cell_type == "code":
        print(f"Executing cell {cell_index}...", flush=True)
client = NotebookClient(nb, kernel_name="iscas2027", timeout=600, on_cell_start=progress)
client.execute(cwd=str(Path.cwd()))
nbformat.write(nb, p)
print("Notebook completed and saved.", flush=True)
"""
        subprocess.run([str(PYTHON), "-c", script], check=True, cwd=ROOT)
    else:
        subprocess.run([str(PYTHON), "-m", "jupyterlab", NOTEBOOK], check=True, cwd=ROOT)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Setup or execution failed: {error}", file=sys.stderr)
        sys.exit(1)
