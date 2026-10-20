#!/usr/bin/env python3
"""Validate databricks.yml and its included resource files against the bundle JSON schema.

`databricks bundle validate` needs workspace credentials; this offline check
catches misspelled or misplaced fields in CI without them. Usage:

    databricks bundle schema > schema.json
    python scripts/check_bundle_schema.py schema.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import jsonschema
import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    schema = _python_patterns(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    validator = jsonschema.Draft7Validator(schema)

    root_config = yaml.safe_load((ROOT / "databricks.yml").read_text(encoding="utf-8"))
    files = [ROOT / "databricks.yml"]
    for pattern in root_config.get("include", []):
        files.extend(sorted(ROOT.glob(pattern)))

    errors = 0
    for path in files:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for error in validator.iter_errors(config):
            if error.context:
                error = jsonschema.exceptions.best_match(error.context)
            location = "/".join(str(part) for part in error.absolute_path)
            print(f"{path.relative_to(ROOT)}: {location}: {error.message}")
            errors += 1
    if errors:
        raise SystemExit(f"{errors} bundle schema error(s)")
    print(f"Bundle configuration matches the schema ({len(files)} files)")


def _python_patterns(node: object) -> object:
    """Rewrite the schema's Go regex classes (``\\p{L}``, ``\\p{N}``) for Python's ``re``.

    Variable names in bundle files are ASCII, so ASCII ranges are enough.
    """

    if isinstance(node, dict):
        output = {key: _python_patterns(value) for key, value in node.items()}
        pattern = output.get("pattern")
        if isinstance(pattern, str):
            output["pattern"] = (
                pattern.replace("[\\p{L}\\p{N}]", "[A-Za-z0-9]")
                .replace("\\p{L}", "[A-Za-z]")
                .replace("\\p{N}", "[0-9]")
            )
            re.compile(output["pattern"])
        return output
    if isinstance(node, list):
        return [_python_patterns(value) for value in node]
    return node

if __name__ == "__main__":
    main()
