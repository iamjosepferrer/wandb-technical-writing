#!/usr/bin/env bash
# setup.sh
#
# Installs Python dependencies for parse_sessions.py.
# Run once after cloning the repo.
#
# Usage:
#   bash setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ok()    { echo "  ✓  $*"; }
info()  { echo "  →  $*"; }
error() { echo "  ✗  $*" >&2; exit 1; }

echo ""
echo "OpenClaw + W&B Weave cost tracker setup"
echo "========================================"
echo ""

command -v python3 &>/dev/null || error "python3 not found. Install Python 3.10+ first."
ok "python3: $(python3 --version)"

# Create venv inside the project directory
VENV="${SCRIPT_DIR}/.venv"

if [ -d "${VENV}" ]; then
    info "Virtual environment exists, updating..."
else
    info "Creating virtual environment..."
    python3 -m venv "${VENV}"
    ok "Created at ${VENV}"
fi

info "Installing dependencies..."
"${VENV}/bin/pip" install --quiet --upgrade pip
"${VENV}/bin/pip" install --quiet -r "${SCRIPT_DIR}/requirements.txt"
"${VENV}/bin/python" -c "import weave, wandb" || error "Install verification failed."
ok "wandb and weave installed"

# Check WANDB_API_KEY
if [ -n "${WANDB_API_KEY:-}" ]; then
    ok "WANDB_API_KEY already in environment"
elif grep -q "^WANDB_API_KEY=" "${HOME}/.openclaw/env" 2>/dev/null; then
    ok "WANDB_API_KEY found in ~/.openclaw/env"
else
    echo ""
    echo "Enter your Weights & Biases API key (from wandb.ai/settings):"
    read -r -p "  WANDB_API_KEY: " KEY
    if [ -n "${KEY}" ]; then
        echo "WANDB_API_KEY=${KEY}" >> "${HOME}/.openclaw/env"
        export WANDB_API_KEY="${KEY}"
        ok "Saved to ~/.openclaw/env"
    else
        echo "  ⚠  Skipped. Set WANDB_API_KEY before running parse_sessions.py"
    fi
fi

echo ""
echo "Setup complete."
echo ""
echo "Run a few conversations in Telegram, then ship them to Weave:"
echo ""
echo "  ${VENV}/bin/python ${SCRIPT_DIR}/parse_sessions.py"
echo ""
echo "Options:"
echo "  --days 7          only include sessions from the last 7 days"
echo "  --dry-run         parse and print without sending to Weave"
echo "  --project NAME    use a custom Weave project name"
echo ""
