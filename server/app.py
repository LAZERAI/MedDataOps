from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import uvicorn

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_SERVER_PATH = ROOT_DIR / "server.py"

spec = importlib.util.spec_from_file_location("meddataops_root_server", ROOT_SERVER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load root server module from {ROOT_SERVER_PATH}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

app = module.app


def create_app():
    return app


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
