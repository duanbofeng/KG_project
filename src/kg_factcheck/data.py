"""Parsing and formatting for the SWC/GERBIL fact-checking data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


RDF_SUBJECT = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
RDF_PREDICATE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"
RDF_OBJECT = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object"
HAS_TRUTH_VALUE = "http://swc2017.aksw.org/hasTruthValue"
XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"

_TRIPLE_RE = re.compile(r"^<([^>]+)>\s+<([^>]+)>\s+(.+?)\s+\.\s*$")
_URI_RE = re.compile(r"^<([^>]+)>$")
_FLOAT_RE = re.compile(r'"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"\^\^<[^>]+>')


@dataclass(frozen=True)
class Fact:
    """A reified RDF statement from the project data."""

    fact_uri: str
    subject: str
    predicate: str
    object: str
    truth: float | None = None


def parse_facts(path: str | Path, require_truth: bool | None = None) -> list[Fact]:
    """Parse train/test TTL-like files containing reified RDF statements."""

    path = Path(path)
    raw: dict[str, dict[str, str | float]] = {}

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        match = _TRIPLE_RE.match(line)
        if not match:
            raise ValueError(f"{path}:{line_number}: invalid TTL line: {line}")

        fact_uri, prop_uri, raw_value = match.groups()
        fields = raw.setdefault(fact_uri, {})

        if prop_uri == RDF_SUBJECT:
            fields["subject"] = _parse_uri_value(raw_value, path, line_number)
        elif prop_uri == RDF_PREDICATE:
            fields["predicate"] = _parse_uri_value(raw_value, path, line_number)
        elif prop_uri == RDF_OBJECT:
            fields["object"] = _parse_uri_value(raw_value, path, line_number)
        elif prop_uri == HAS_TRUTH_VALUE:
            value_match = _FLOAT_RE.search(raw_value)
            if not value_match:
                raise ValueError(f"{path}:{line_number}: invalid truth value: {raw_value}")
            fields["truth"] = float(value_match.group(1))

    facts: list[Fact] = []
    for fact_uri, fields in raw.items():
        missing = {"subject", "predicate", "object"} - fields.keys()
        if missing:
            raise ValueError(f"{path}: fact <{fact_uri}> is missing {sorted(missing)}")
        truth = fields.get("truth")
        if require_truth is True and truth is None:
            raise ValueError(f"{path}: fact <{fact_uri}> is missing a truth value")
        if require_truth is False and truth is not None:
            raise ValueError(f"{path}: fact <{fact_uri}> unexpectedly has a truth value")
        facts.append(
            Fact(
                fact_uri=fact_uri,
                subject=str(fields["subject"]),
                predicate=str(fields["predicate"]),
                object=str(fields["object"]),
                truth=None if truth is None else float(truth),
            )
        )
    return facts


def _parse_uri_value(value: str, path: Path, line_number: int) -> str:
    match = _URI_RE.match(value)
    if not match:
        raise ValueError(f"{path}:{line_number}: expected URI object, got: {value}")
    return match.group(1)


def local_name(uri: str) -> str:
    """Return a compact readable name for a URI."""

    return re.split(r"[/#]", uri.rstrip("/#"))[-1]

