"""
Step 0: Verify your environment.

Checks Python version, required packages, and the .env file.
No API calls are made here — safe to run before adding any keys.

Usage:
    python step0_verify_env.py
"""

import importlib
import os
import sys

REQUIRED_PACKAGES = [
    "langchain_core",
    "langchain_text_splitters",
    "langchain_openai",
    "langchain_chroma",
    "chromadb",
    "openai",
    "weave",
    "wandb",
    "dotenv",
    "tiktoken",
]

REQUIRED_ENV_VARS = ["OPENAI_API_KEY"]


def check_python_version() -> bool:
    version = sys.version_info
    ok = version >= (3, 9)
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] Python {version.major}.{version.minor}.{version.micro}  (need >= 3.9, recommend 3.11)")
    return ok


def check_packages() -> bool:
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"  [OK]   {pkg:<20}  {version}")
        except ImportError:
            print(f"  [FAIL] {pkg:<20}  not installed")
            all_ok = False
    return all_ok


def check_env_file() -> bool:
    """Check .env exists and contains expected keys (values are not validated)."""
    if not os.path.exists(".env"):
        print("  [WARN] .env file not found.")
        print("         Copy .env.example to .env and fill in your keys.")
        print("         Alternatively run: wandb login")
        return False

    with open(".env") as f:
        content = f.read()

    all_ok = True
    for var in REQUIRED_ENV_VARS:
        if var in content:
            print(f"  [OK]   {var} found in .env")
        else:
            print(f"  [WARN] {var} not found in .env")
            all_ok = False

    # WANDB_API_KEY is optional if the user ran "wandb login"
    if "WANDB_API_KEY" in content:
        print("  [OK]   WANDB_API_KEY found in .env")
    else:
        print("  [INFO] WANDB_API_KEY not in .env — that is fine if you ran 'wandb login'")

    return all_ok


def main() -> None:
    print("\n=== Step 0: Environment check ===\n")

    print("Python version:")
    py_ok = check_python_version()

    print("\nRequired packages:")
    pkg_ok = check_packages()

    print("\n.env file:")
    check_env_file()

    print()
    if py_ok and pkg_ok:
        print("Environment looks good. Run step1_knowledge_base.py next.")
    else:
        print("Fix the issues above before continuing.")
        if not py_ok:
            print("  Upgrade to Python 3.9 or later (3.11 recommended).")
        if not pkg_ok:
            print("  Run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
