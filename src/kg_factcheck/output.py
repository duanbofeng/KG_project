"""GERBIL result writing and validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from .data import Fact, HAS_TRUTH_VALUE, XSD_DOUBLE


RESULT_RE = re.compile(
    r'^<([^>]+)>\s+<http://swc2017\.aksw\.org/hasTruthValue>\s+'
    r'"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"\^\^<http://www\.w3\.org/2001/XMLSchema#double>\s+\.\s*$'
)


def write_result(path: str | Path, facts: list[Fact], scores: list[float]) -> None:
    if len(facts) != len(scores):
        raise ValueError("facts and scores must have the same length")

    lines = []
    for fact, score in zip(facts, scores):
        clipped = min(1.0, max(0.0, float(score)))
        lines.append(
            f'<{fact.fact_uri}> <{HAS_TRUTH_VALUE}> "{clipped:.10f}"^^<{XSD_DOUBLE}> .'
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_result(path: str | Path, test_facts: list[Fact]) -> tuple[bool, list[str]]:
    expected = [fact.fact_uri for fact in test_facts]
    expected_set = set(expected)
    seen: list[str] = []
    errors: list[str] = []

    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        match = RESULT_RE.match(line)
        if not match:
            errors.append(f"line {line_number}: invalid GERBIL result format")
            continue
        fact_uri, raw_score = match.groups()
        score = float(raw_score)
        if not 0.0 <= score <= 1.0:
            errors.append(f"line {line_number}: score outside [0,1]: {score}")
        if fact_uri not in expected_set:
            errors.append(f"line {line_number}: unexpected fact URI <{fact_uri}>")
        seen.append(fact_uri)

    counts = Counter(seen)
    duplicates = sorted(uri for uri, count in counts.items() if count > 1)
    missing = sorted(expected_set - set(seen))
    if duplicates:
        errors.append(f"duplicate predictions: {len(duplicates)}")
    if missing:
        errors.append(f"missing predictions: {len(missing)}")
    if len(seen) != len(expected):
        errors.append(f"wrong number of predictions: got {len(seen)}, expected {len(expected)}")

    return not errors, errors

