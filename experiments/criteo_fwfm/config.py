from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration parsing or validation fails."""


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top-level YAML must be a mapping: {path}")
    return loaded


def set_nested_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not all(parts):
        raise ConfigError(f"Invalid override key: {dotted_key!r}")
    cursor: dict[str, Any] = config
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if next_value is None:
            next_value = {}
            cursor[part] = next_value
        if not isinstance(next_value, dict):
            raise ConfigError(
                f"Cannot set {dotted_key!r}: {part!r} is not a mapping in current config"
            )
        cursor = next_value
    cursor[parts[-1]] = value


def _parse_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise ConfigError(
            f"Override must be in key=value form, got: {override!r}"
        )
    key, raw_value = override.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"Override key cannot be empty: {override!r}")
    value = yaml.safe_load(raw_value)
    return key, value


def resolve_config(
    default_config_path: Path,
    config_paths: list[Path],
    overrides: list[str],
) -> dict[str, Any]:
    config = load_yaml(default_config_path)
    for path in config_paths:
        config = _deep_merge(config, load_yaml(path))

    for override in overrides:
        key, value = _parse_override(override)
        set_nested_key(config, key, value)
    return config


def get_config_value(
    config: dict[str, Any],
    dotted_key: str,
    *,
    default: Any | None = None,
    required: bool = False,
) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if isinstance(cursor, dict) and part in cursor:
            cursor = cursor[part]
        else:
            if required:
                raise ConfigError(f"Missing required config key: {dotted_key}")
            return default
    return cursor


def save_yaml(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
