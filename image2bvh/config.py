"""User-editable configuration backed by ``config.ini`` at the project root.

Layout (auto-generated on first run; missing keys are filled in with
defaults the next time it is loaded so adding new options is safe):

    [ui]
    language = ja          ; "ja" or "en"

    [models]
    sam3_repo_id        = facebook/sam3
    sam3dbody_repo_id   = jetjodh/sam-3d-body-dinov3

Older builds also tracked Blender Portable URLs and a BVH backend
toggle (``[blender]`` / ``[bvh] backend``). Those sections were
removed when Blender support was dropped — BVH is now always emitted
by the pure-Python writer in ``image2bvh/bvh_writer.py``. Existing
sections in your config.ini are simply ignored; you can delete them
by hand if you want.

Repo IDs are settings, not code, so HuggingFace renames can be
tracked by editing config.ini without touching the source.
"""
from __future__ import annotations

import configparser
from pathlib import Path

from . import paths

CONFIG_PATH: Path = paths.PROJECT_ROOT / "config.ini"

DEFAULTS: dict[str, dict[str, str]] = {
    "ui": {
        "language": "ja",
        # Show the inference log panel in the WebUI ("true"/"false").
        # Hide it for a cleaner end-user surface — the same lines still
        # appear on the server-side console regardless of this setting.
        "show_log": "true",
    },
    "models": {
        "sam3_repo_id": "facebook/sam3",
        "sam3dbody_repo_id": "jetjodh/sam-3d-body-dinov3",
    },
}

VALID_LANGS = ("ja", "en")


def _new_parser() -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    cp.optionxform = str  # preserve key case
    return cp


def load() -> configparser.ConfigParser:
    """Read config.ini, filling in defaults for any missing section/key.

    Side effect: if anything was missing (or the file did not exist), the
    merged result is written back to disk so future edits start from a
    complete file.
    """
    cp = _new_parser()
    if CONFIG_PATH.is_file():
        try:
            cp.read(CONFIG_PATH, encoding="utf-8")
        except configparser.Error:
            # corrupt file — treat as empty, defaults will repopulate it
            cp = _new_parser()

    changed = False
    for section, kvs in DEFAULTS.items():
        if section not in cp:
            cp[section] = {}
            changed = True
        for k, v in kvs.items():
            if k not in cp[section] or cp[section][k] == "":
                cp[section][k] = v
                changed = True

    # Validate language
    lang = cp["ui"].get("language", "ja").strip().lower()
    if lang not in VALID_LANGS:
        cp["ui"]["language"] = "ja"
        changed = True

    if changed:
        save(cp)
    return cp


def save(cp: configparser.ConfigParser) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        cp.write(f)


def get(section: str, key: str) -> str:
    return load()[section][key]


def set_value(section: str, key: str, value: str) -> None:
    cp = load()
    if section not in cp:
        cp[section] = {}
    cp[section][key] = value
    save(cp)


def language() -> str:
    return load()["ui"]["language"].strip().lower()


def set_language(lang: str) -> None:
    if lang not in VALID_LANGS:
        raise ValueError(f"unsupported language: {lang}")
    set_value("ui", "language", lang)


_TRUTHY = ("true", "1", "yes", "on")


def show_log() -> bool:
    raw = load()["ui"].get("show_log", "true").strip().lower()
    return raw in _TRUTHY
