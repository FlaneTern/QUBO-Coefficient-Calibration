# setup.py
from venv import create as create_venv
from os import getcwd, name as os_name
from os.path import join
from subprocess import run
from pathlib import Path
import sys


def main():
    gpu = "--gpu" in sys.argv
    if gpu:
        sys.argv.remove("--gpu")
    if len(sys.argv) != 1:
        raise ValueError("Supported usage: python setup.py [--gpu]")
    if gpu and os_name == "nt":
        raise ValueError("The project GPU preset is intended for Linux/NVIDIA CUDA 12; use CPU setup on Windows")

    venv_dir = join(getcwd(), "venv")
    print(f"Creating virtual environment at: {venv_dir}")
    create_venv(venv_dir, with_pip=True)

    scripts_dir = "Scripts" if os_name == "nt" else "bin"
    python_exe = join(venv_dir, scripts_dir,
                      "python.exe" if os_name == "nt" else "python")

    base_dir = Path(__file__).resolve().parent
    if gpu:
        req_file = base_dir / "requirements-gpu.txt"
    else:
        req_file = base_dir / "requirements-cpu.txt"

    if not req_file.exists():
        raise FileNotFoundError(f"{req_file.name} not found at {req_file}")

    print(f"Detected OS: {'Windows' if os_name == 'nt' else 'Linux/Unix-like'}")
    print(f"Compute preset: {'GPU-assisted (NVIDIA/CUDA 12)' if gpu else 'CPU/default'}")
    print(f"Using requirements file: {req_file}")

    print("Upgrading pip inside venv…")
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    print(f"Installing dependencies from {req_file} ...")
    run([python_exe, "-m", "pip", "install", "-r", str(req_file)], check=True)

    device = "lightning.gpu" if gpu else "lightning.qubit"
    print(f"Validating PennyLane runtime ({device}) ...")
    validation_code = (
        "import pennylane as qml; "
        "dev = qml.device(" + repr(device) + ", wires=1); "
        "print('PennyLane', qml.__version__, 'OK -', type(dev).__name__)"
    )
    run([python_exe, "-c", validation_code], check=True)

    print("\nDone.")
    print("To activate the venv:")
    if os_name == "nt":
        print(rf"  PowerShell: .\venv\Scripts\Activate.ps1")
        print(rf"  CMD      : .\venv\Scripts\activate.bat")
    else:
        print(rf"  Bash/Zsh : source ./venv/bin/activate")
        if gpu:
            print("  GPU preset installed. Run your experiment with: --compute-device gpu")



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Setup failed: {e}", file=sys.stderr)
        sys.exit(1)
