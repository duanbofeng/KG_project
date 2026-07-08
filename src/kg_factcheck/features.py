"""Feature extraction for the fact-checking model."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import re

from .data import Fact, local_name


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def sigmoid(value: float) -> float:
    if value < -35.0:
        return 0.0
    if value > 35.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def smooth_rate(positives: float, total: float, prior: float, strength: float = 2.0) -> float:
    return (positives + prior * strength) / (total + strength)


class FeatureEncoder:
    """Builds sparse hashed features plus smoothed empirical statistics."""

    def __init__(self, num_buckets: int = 4096) -> None:
        self.num_buckets = num_buckets
        self.global_rate = 0.5
        self.predicate_rates: dict[str, float] = {}
        self.subject_rates: dict[str, float] = {}
        self.object_rates: dict[str, float] = {}
        self.subject_counts: dict[str, int] = {}
        self.object_counts: dict[str, int] = {}
        self.triple_truth: dict[str, float] = {}

    def fit(self, facts: list[Fact]) -> None:
        labeled = [fact for fact in facts if fact.truth is not None]
        if not labeled:
            raise ValueError("FeatureEncoder.fit requires labeled training facts")

        total_positive = sum(float(fact.truth) for fact in labeled)
        self.global_rate = smooth_rate(total_positive, len(labeled), 0.5, strength=2.0)
        self.predicate_rates = self._rates_by(labeled, lambda fact: fact.predicate, strength=4.0)
        self.subject_rates = self._rates_by(labeled, lambda fact: fact.subject, strength=6.0)
        self.object_rates = self._rates_by(labeled, lambda fact: fact.object, strength=6.0)
        self.subject_counts = dict(Counter(fact.subject for fact in labeled))
        self.object_counts = dict(Counter(fact.object for fact in labeled))
        self.triple_truth = {
            self._triple_key(fact): float(fact.truth)
            for fact in labeled
            if fact.truth is not None
        }

    def transform(self, fact: Fact) -> dict[int, float]:
        features: dict[int, float] = {}

        self._add(features, "bias", 1.0)
        self._add(features, f"pred={fact.predicate}", 1.0)
        self._add(features, f"subject_seen={fact.subject in self.subject_counts}", 1.0)
        self._add(features, f"object_seen={fact.object in self.object_counts}", 1.0)

        predicate_name = local_name(fact.predicate)
        subject_tokens = tokenize(local_name(fact.subject))
        object_tokens = tokenize(local_name(fact.object))
        predicate_tokens = tokenize(predicate_name)

        for token in predicate_tokens:
            self._add(features, f"pred_token={token}", 1.0)
        for token in subject_tokens:
            self._add(features, f"subj_token={token}", 1.0)
            self._add(features, f"pred_subj_token={predicate_name}:{token}", 1.0)
        for token in object_tokens:
            self._add(features, f"obj_token={token}", 1.0)
            self._add(features, f"pred_obj_token={predicate_name}:{token}", 1.0)

        overlap = len(set(subject_tokens) & set(object_tokens))
        self._add_numeric(features, "predicate_rate", self.predicate_rates.get(fact.predicate, self.global_rate))
        self._add_numeric(features, "subject_rate", self.subject_rates.get(fact.subject, self.global_rate))
        self._add_numeric(features, "object_rate", self.object_rates.get(fact.object, self.global_rate))
        self._add_numeric(features, "subject_count", math.log1p(self.subject_counts.get(fact.subject, 0)))
        self._add_numeric(features, "object_count", math.log1p(self.object_counts.get(fact.object, 0)))
        self._add_numeric(features, "token_overlap", float(overlap))
        self._add_numeric(features, "subject_len", min(len(subject_tokens), 12) / 12.0)
        self._add_numeric(features, "object_len", min(len(object_tokens), 12) / 12.0)

        return features

    def exact_truth(self, fact: Fact) -> float | None:
        return self.triple_truth.get(self._triple_key(fact))

    def to_dict(self) -> dict[str, object]:
        return {
            "num_buckets": self.num_buckets,
            "global_rate": self.global_rate,
            "predicate_rates": self.predicate_rates,
            "subject_rates": self.subject_rates,
            "object_rates": self.object_rates,
            "subject_counts": self.subject_counts,
            "object_counts": self.object_counts,
            "triple_truth": self.triple_truth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "FeatureEncoder":
        encoder = cls(num_buckets=int(data["num_buckets"]))
        encoder.global_rate = float(data["global_rate"])
        encoder.predicate_rates = {str(k): float(v) for k, v in dict(data["predicate_rates"]).items()}
        encoder.subject_rates = {str(k): float(v) for k, v in dict(data["subject_rates"]).items()}
        encoder.object_rates = {str(k): float(v) for k, v in dict(data["object_rates"]).items()}
        encoder.subject_counts = {str(k): int(v) for k, v in dict(data["subject_counts"]).items()}
        encoder.object_counts = {str(k): int(v) for k, v in dict(data["object_counts"]).items()}
        encoder.triple_truth = {str(k): float(v) for k, v in dict(data["triple_truth"]).items()}
        return encoder

    def _rates_by(self, facts: list[Fact], key_fn, strength: float) -> dict[str, float]:
        positives: defaultdict[str, float] = defaultdict(float)
        totals: defaultdict[str, int] = defaultdict(int)
        for fact in facts:
            key = key_fn(fact)
            positives[key] += float(fact.truth)
            totals[key] += 1
        return {
            key: smooth_rate(positives[key], totals[key], self.global_rate, strength=strength)
            for key in totals
        }

    def _add(self, features: dict[int, float], name: str, value: float) -> None:
        index = stable_hash(name, self.num_buckets)
        features[index] = features.get(index, 0.0) + value

    def _add_numeric(self, features: dict[int, float], name: str, value: float) -> None:
        self._add(features, f"num:{name}", value)

    @staticmethod
    def _triple_key(fact: Fact) -> str:
        return f"{fact.subject}\t{fact.predicate}\t{fact.object}"


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text.replace("_", " "))]


def stable_hash(value: str, buckets: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets

