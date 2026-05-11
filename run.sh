#!/usr/bin/env bash
# Launcher for Linux (x64 / aarch64) and macOS (arm64 / x64).
# Usage:
#   ./run.sh                 # launch UI
#   ./run.sh --share         # passes through to the app
set -euo pipefail
cd "$(dirname "$0")"

# --- 1) ensure uv -----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[image2BVH] 'uv' not found on PATH. Installing uv ..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "[image2BVH] need curl or wget to install uv. Install one and retry,"
        echo "            or install uv manually: https://docs.astral.sh/uv/"
        exit 1
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# --- 2) platform hint -------------------------------------------------------
case "$(uname -s)" in
    Darwin)
        echo "[image2BVH] macOS detected. PyTorch will use CPU/MPS (no CUDA)."
        ;;
    Linux)
        case "$(uname -m)" in
            aarch64)
                echo "[image2BVH] Linux aarch64 detected (DGX Spark / GB10 build path)."
                echo "             torch=cu130 nightly."
                ;;
            x86_64)
                echo "[image2BVH] Linux x86_64 detected. torch=cu128 stable."
                ;;
        esac
        ;;
esac

# --- 3) sync deps -----------------------------------------------------------
echo "[image2BVH] Syncing dependencies (this can take a while on first run) ..."
uv sync --no-dev

# --- 4) HuggingFace login check (SAM 3 is gated) ----------------------------
check_hf_auth() {
    if [ -f "runtime/models/sam3/config.json" ]; then
        echo "[image2BVH] SAM 3 model already present locally, skipping HF login prompt."
        return
    fi
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "[image2BVH] HF_TOKEN already set in environment, skipping login prompt."
        return
    fi
    if uv run --no-dev huggingface-cli whoami >/dev/null 2>&1; then
        return  # already logged in via huggingface-cli login
    fi
    cat <<'BANNER'

=====================================================================
 SAM 3 (used for person mask generation) is GATED on HuggingFace.

   1. Apply for access at  https://huggingface.co/facebook/sam3
      (manual approval by Meta — usually a few hours to a few days)
   2. Create a read token at  https://huggingface.co/settings/tokens
   3. Paste the token below (or press Enter to skip and configure later)
=====================================================================
BANNER
    # -s would hide the typed token; commented out so users can verify
    # they pasted correctly. Uncomment if you prefer silent entry.
    # read -r -s -p "HF_TOKEN: " HFTOKEN; echo
    read -r -p "HF_TOKEN: " HFTOKEN || HFTOKEN=""
    if [ -n "$HFTOKEN" ]; then
        echo "[image2BVH] Saving HF token via huggingface-cli login ..."
        uv run --no-dev huggingface-cli login --token "$HFTOKEN"
        unset HFTOKEN
    else
        echo "[image2BVH] Skipped. Without a valid token, SAM 3 download will fail"
        echo "            until you set HF_TOKEN or run 'huggingface-cli login'."
    fi
}
check_hf_auth

# --- 5) launch --------------------------------------------------------------
echo "[image2BVH] Launching ..."
exec uv run --no-dev python -m image2bvh "$@"
