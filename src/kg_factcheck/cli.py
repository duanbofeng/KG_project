"""Command line interface for the KG fact-checking project."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from .data import Fact, local_name, parse_facts
from .model import FactCheckModel, TrainingConfig, cross_validate_auc
from .output import validate_result, write_result
from .sparql import SparqlEvidenceCache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kg-factcheck")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a local fact-checking model")
    train_parser.add_argument("--train", default="train.txt")
    train_parser.add_argument("--model", default="model.joblib")
    train_parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    train_parser.add_argument("--buckets", type=int, default=TrainingConfig.num_buckets)

    predict_parser = subparsers.add_parser("predict", help="predict GERBIL truth scores")
    predict_parser.add_argument("--test", default="test.txt")
    predict_parser.add_argument("--model", default="model.joblib")
    predict_parser.add_argument("--output", default="result.ttl")
    predict_parser.add_argument("--use-sparql", action="store_true", help="blend optional DBpedia exact-triple evidence")
    predict_parser.add_argument("--sparql-weight", type=float, default=0.15)

    validate_parser = subparsers.add_parser("validate", help="validate a GERBIL result file")
    validate_parser.add_argument("--result", default="result.ttl")
    validate_parser.add_argument("--test", default="test.txt")

    analyze_parser = subparsers.add_parser("analyze", help="summarize data and run local cross validation")
    analyze_parser.add_argument("--train", default="train.txt")
    analyze_parser.add_argument("--test", default="test.txt")
    analyze_parser.add_argument("--folds", type=int, default=5)
    analyze_parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)

    args = parser.parse_args(argv)

    if args.command == "train":
        return _train(args)
    if args.command == "predict":
        return _predict(args)
    if args.command == "validate":
        return _validate(args)
    if args.command == "analyze":
        return _analyze(args)
    parser.error(f"unknown command: {args.command}")
    return 2


def _train(args: argparse.Namespace) -> int:
    facts = parse_facts(args.train, require_truth=True)
    config = TrainingConfig(epochs=args.epochs, num_buckets=args.buckets)
    model = FactCheckModel.train(facts, config=config)
    model.save(args.model)
    print(f"trained on {len(facts)} facts; saved model to {args.model}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    facts = parse_facts(args.test, require_truth=False)
    model = FactCheckModel.load(args.model)
    scores = model.predict(facts)

    if args.use_sparql:
        cache = SparqlEvidenceCache()
        blended = []
        weight = min(1.0, max(0.0, args.sparql_weight))
        for fact, score in zip(facts, scores):
            evidence = cache.exact_triple_score(fact)
            blended.append((1.0 - weight) * score + weight * evidence)
        scores = blended

    write_result(args.output, facts, scores)
    ok, errors = validate_result(args.output, facts)
    if not ok:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"wrote {len(facts)} predictions to {args.output}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    facts = parse_facts(args.test, require_truth=False)
    ok, errors = validate_result(args.result, facts)
    if ok:
        print(f"valid GERBIL result file: {args.result} ({len(facts)} predictions)")
        return 0
    for error in errors:
        print(error, file=sys.stderr)
    return 1


def _analyze(args: argparse.Namespace) -> int:
    train_facts = parse_facts(args.train, require_truth=True)
    test_facts = parse_facts(args.test, require_truth=False) if Path(args.test).exists() else []

    print(f"train facts: {len(train_facts)}")
    print(f"test facts: {len(test_facts)}")
    print(f"train positive rate: {_positive_rate(train_facts):.4f}")
    _print_predicate_summary("train predicate summary", train_facts)
    if test_facts:
        _print_predicate_summary("test predicate summary", test_facts)

    config = TrainingConfig(epochs=args.epochs)
    cv = cross_validate_auc(train_facts, folds=args.folds, config=config)
    fold_text = ", ".join(f"{value:.4f}" for value in cv["folds"])
    print(f"{args.folds}-fold ROC AUC: mean={cv['mean']:.4f}, pooled={cv['pooled']:.4f}, folds=[{fold_text}]")
    return 0


def _positive_rate(facts: list[Fact]) -> float:
    labels = [float(fact.truth) for fact in facts if fact.truth is not None]
    return sum(labels) / len(labels)


def _print_predicate_summary(title: str, facts: list[Fact]) -> None:
    counts = Counter(fact.predicate for fact in facts)
    print(title + ":")
    for predicate, count in counts.most_common():
        labeled = [fact for fact in facts if fact.predicate == predicate and fact.truth is not None]
        if labeled:
            rate = sum(float(fact.truth) for fact in labeled) / len(labeled)
            print(f"  {local_name(predicate):16s} {count:4d} positive_rate={rate:.3f}")
        else:
            print(f"  {local_name(predicate):16s} {count:4d}")


if __name__ == "__main__":
    raise SystemExit(main())

