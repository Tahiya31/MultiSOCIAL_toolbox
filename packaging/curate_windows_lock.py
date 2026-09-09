"""Create audited, hashed GUI/worker locks from a pip-compile Windows lock."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==")

def canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def split_lock(text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    preamble: list[str] = []
    blocks: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        match = REQUIREMENT.match(line)
        if match:
            if current_name is not None:
                blocks.append((current_name, current_lines))
            elif current_lines:
                preamble.extend(current_lines)
            current_name = canonicalize_name(match.group(1))
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_name is not None:
        blocks.append((current_name, current_lines))
    else:
        preamble.extend(current_lines)
    return preamble, blocks


def curated_text(
    source: Path,
    *,
    excluded: set[str] | None = None,
    included: set[str] | None = None,
    title: str,
) -> str:
    preamble, blocks = split_lock(source.read_text(encoding="utf-8"))
    excluded = {canonicalize_name(name) for name in excluded or set()}
    included = {canonicalize_name(name) for name in included or set()}
    selected = [
        lines
        for name, lines in blocks
        if name not in excluded and (not included or name in included)
    ]
    names = {
        canonicalize_name(REQUIREMENT.match(lines[0]).group(1))
        for lines in selected
        if REQUIREMENT.match(lines[0])
    }
    if included and names != included:
        missing = sorted(included - names)
        raise RuntimeError("Packages missing from source lock: " + ", ".join(missing))
    header = [
        f"# {title}\n",
        "# Generated from the committed pip-compile lock; every installed file remains hash-checked.\n",
        "# Install with: pip install --require-hashes --no-deps -r <this-file>\n",
        "\n",
    ]
    return "".join([*header, *[line for block in selected for line in block]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    args.output.write_text(
        curated_text(
            args.source,
            excluded=set(args.exclude),
            included=set(args.include),
            title=args.title,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
