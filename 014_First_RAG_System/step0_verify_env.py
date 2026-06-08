"""
Step 0: Verify the environment before running the tutorial.

Run this first. If anything fails, fix it before moving to step 1.

Usage:
    python step0_verify_env.py
"""

import importlib
import os
import sys


def check_python_version() -> bool:
    v = sys.version_info
    print(f"  Python {v.major}.{v.minor}.{v.micro}")
    if v.major != 3 or v.minor < 10:
        print("  WARNING: Python 3.10+ required (tutorial tested on 3.11.x)")
        return False
    return True


def check_packages() -> bool:
    # (import_name, display_name)
    packages = [
        ("openai", "openai"),
        ("weave", "weave"),
        ("faiss", "faiss-cpu"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("tiktoken", "tiktoken"),
        ("dotenv", "python-dotenv"),
    ]

    all_ok = True
    for import_name, display_name in packages:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {display_name}: {version}")
        except ImportError:
            print(f"  MISSING: {display_name}  ->  pip install -r requirements.txt")
            all_ok = False
    return all_ok


def check_env_vars() -> bool:
    from dotenv import load_dotenv

    load_dotenv()

    required = ["OPENAI_API_KEY"]
    optional = ["WANDB_ENTITY"]

    all_ok = True
    for var in required:
        val = os.getenv(var)
        if val:
            # Show only the last 4 characters to confirm it's set
            print(f"  {var}: set (***{val[-4:]})")
        else:
            print(f"  {var}: NOT SET  ->  add to your .env file")
            all_ok = False

    for var in optional:
        val = os.getenv(var)
        status = val if val else "not set (will use default W&B entity)"
        print(f"  {var}: {status}")

    return all_ok


def main() -> None:
    print("=" * 50)
    print("Environment check")
    print("=" * 50)

    print("\nPython version:")
    python_ok = check_python_version()

    print("\nPackages:")
    packages_ok = check_packages()

    print("\nEnvironment variables:")
    env_ok = check_env_vars()

    print()
    if python_ok and packages_ok and env_ok:
        print("All checks passed. Proceed to step1_knowledge_base.py")
    else:
        print("Fix the issues above before running the rest of the tutorial.")
        sys.exit(1)


if __name__ == "__main__":
    main()
