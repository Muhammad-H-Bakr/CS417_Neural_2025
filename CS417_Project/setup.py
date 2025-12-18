import subprocess
import sys
import platform
from pathlib import Path

# Configuration
VENV_DIR = ".venv"  # virtual environment folder
REQ_FILE = "requirements.txt"
KERNEL_NAME = "myenv"  # kernel name for Jupyter


def create_venv():
    if not Path(VENV_DIR).exists():
        print(f"Creating virtual environment in {VENV_DIR}...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print(f"Virtual environment already exists at {VENV_DIR}")


def get_pip_path():
    if platform.system() == "Windows":
        return Path(VENV_DIR) / "Scripts" / "pip.exe"
    else:
        return Path(VENV_DIR) / "bin" / "pip"


def install_requirements():
    pip_path = str(get_pip_path())
    if not Path(REQ_FILE).exists():
        print(f" {REQ_FILE} not found.")
        return
    print(f" Installing packages from {REQ_FILE}...")
    subprocess.check_call([pip_path, "install", "-r", REQ_FILE])
    print("All packages installed!")


def setup_jupyter_kernel():
    pip_path = str(get_pip_path())
    print("Setting up Jupyter kernel...")
    subprocess.check_call([pip_path, "install", "ipykernel"])
    python_path = (
        Path(VENV_DIR)
        / ("Scripts" if platform.system() == "Windows" else "bin")
        / "python"
    )
    subprocess.check_call(
        [
            str(python_path),
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            KERNEL_NAME,
            "--display-name",
            f"Python ({KERNEL_NAME})",
        ]
    )
    print(f"Jupyter kernel '{KERNEL_NAME}' installed!")


if __name__ == "__main__":
    create_venv()
    install_requirements()
    setup_jupyter_kernel()
    print("Environment setup complete!")
    if platform.system() == "Windows":
        print(f"To activate the venv, run: {VENV_DIR}\\Scripts\\activate")
    else:
        print(f"To activate the venv, run: source {VENV_DIR}/bin/activate")