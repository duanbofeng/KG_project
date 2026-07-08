"""A small dependency-free fact-checking model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path

from .data import Fact
from .features import FeatureEncoder, sigmoid


@dataclass
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 0.08
    l2: float = 0.0002
    seed: int = 13
    num_buckets: int = 4096
    blend_model: float = 0.80
    blend_prior: float = 0.20


class FactCheckModel:
    """Hashed logistic regression blended with predicate priors."""

    def __init__(self, encoder: FeatureEncoder, weights: list[float], config: TrainingConfig) -> None:
        self.encoder = encoder
        self.weights = weights
        self.config = config

    @classmethod
    def train(cls, facts: list[Fact], config: TrainingConfig | None = None) -> "FactCheckModel":
        config = config or TrainingConfig()
        labeled = [fact for fact in facts if fact.truth is not None]
        if not labeled:
            raise ValueError("training requires facts with truth labels")

        encoder = FeatureEncoder(num_buckets=config.num_buckets)
        encoder.fit(labeled)
        weights = [0.0] * config.num_buckets
        rng = random.Random(config.seed)
        rows = [(encoder.transform(fact), float(fact.truth)) for fact in labeled]

        for epoch in range(config.epochs):
            rng.shuffle(rows)
            rate = config.learning_rate / (1.0 + epoch * 0.03)
            for features, label in rows:
                score = sum(weights[index] * value for index, value in features.items())
                pred = sigmoid(score)
                error = pred - label
                for index, value in features.items():
                    gradient = error * value + config.l2 * weights[index]
                    weights[index] -= rate * gradient

        return cls(encoder=encoder, weights=weights, config=config)

    def predict_one(self, fact: Fact) -> float:
        exact = self.encoder.exact_truth(fact)
        if exact is not None:
            return exact

        features = self.encoder.transform(fact)
        raw = sum(self.weights[index] * value for index, value in features.items())
        model_score = sigmoid(raw)
        prior_score = self.encoder.predicate_rates.get(fact.predicate, self.encoder.global_rate)
        score = self.config.blend_model * model_score + self.config.blend_prior * prior_score
        return min(1.0, max(0.0, score))

    def predict(self, facts: list[Fact]) -> list[float]:
        return [self.predict_one(fact) for fact in facts]

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "config": self.config.__dict__,
            "encoder": self.encoder.to_dict(),
            "weights": self.weights,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FactCheckModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = TrainingConfig(**payload["config"])
        encoder = FeatureEncoder.from_dict(payload["encoder"])
        weights = [float(value) for value in payload["weights"]]
        return cls(encoder=encoder, weights=weights, config=config)


def roc_auc(labels: list[float], scores: list[float]) -> float:
    """Compute ROC AUC with average ranks for tied scores."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = sum(1 for label in labels if label >= 0.5)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both positive and negative labels")

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        rank_sum += average_rank * sum(1 for _, label in pairs[index:end] if label >= 0.5)
        rank += end - index
        index = end

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def cross_validate_auc(facts: list[Fact], folds: int = 5, config: TrainingConfig | None = None) -> dict[str, object]:
    labeled = [fact for fact in facts if fact.truth is not None]
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(labeled) < folds:
        raise ValueError("not enough labeled facts for cross validation")

    rng = random.Random((config or TrainingConfig()).seed)
    positives = [fact for fact in labeled if fact.truth and fact.truth >= 0.5]
    negatives = [fact for fact in labeled if not fact.truth or fact.truth < 0.5]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    buckets: list[list[Fact]] = [[] for _ in range(folds)]
    for index, fact in enumerate(positives):
        buckets[index % folds].append(fact)
    for index, fact in enumerate(negatives):
        buckets[index % folds].append(fact)

    aucs: list[float] = []
    all_labels: list[float] = []
    all_scores: list[float] = []
    for fold_index in range(folds):
        validation = buckets[fold_index]
        training = [fact for i, bucket in enumerate(buckets) if i != fold_index for fact in bucket]
        model = FactCheckModel.train(training, config=config)
        scores = model.predict(validation)
        labels = [float(fact.truth) for fact in validation if fact.truth is not None]
        aucs.append(roc_auc(labels, scores))
        all_labels.extend(labels)
        all_scores.extend(scores)

    return {
        "folds": aucs,
        "mean": sum(aucs) / len(aucs),
        "pooled": roc_auc(all_labels, all_scores),
    }
