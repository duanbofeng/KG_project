"""Optional DBpedia SPARQL evidence collection with local caching."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .data import Fact


DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"


class SparqlEvidenceCache:
    """Checks whether exact triples exist in DBpedia and caches results."""

    def __init__(self, cache_path: str | Path = ".kg_cache/dbpedia_exact.json") -> None:
        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            self.cache: dict[str, float] = json.loads(self.cache_path.read_text(encoding="utf-8"))
        else:
            self.cache = {}

    def exact_triple_score(self, fact: Fact, timeout: float = 8.0) -> float:
        key = f"{fact.subject}\t{fact.predicate}\t{fact.object}"
        if key not in self.cache:
            try:
                self.cache[key] = 1.0 if self._ask_exact(fact, timeout=timeout) else 0.0
            except OSError:
                self.cache[key] = 0.5
            self._save()
        return self.cache[key]

    def _ask_exact(self, fact: Fact, timeout: float) -> bool:
        query = f"ASK WHERE {{ <{fact.subject}> <{fact.predicate}> <{fact.object}> }}"
        url = DBPEDIA_ENDPOINT + "?" + urlencode({"query": query, "format": "application/sparql-results+json"})
        request = Request(url, headers={"Accept": "application/sparql-results+json"})
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return bool(payload.get("boolean"))

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8")
