from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(input_path: str | Path, exts: Iterable[str]) -> list[Path]:
    p = Path(input_path)
    exts_l = {e.lower().lstrip(".") for e in exts}
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    files = []
    for ext in exts_l:
        files.extend(p.rglob(f"*.{ext}"))
    return sorted(files)


def get_env_var(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    return None
