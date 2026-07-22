# Knowledge Graph Fact-Checking Mini-Project

This repository implements a reproducible fact-checking engine for the Knowledge Graphs mini-project. It reads reified RDF statements, trains a veracity model on `train.txt`, and writes a GERBIL-compatible `result.ttl` for `test.txt`.

The project goal from the course slides is to return a value between `0` (false) and `1` (true) for each fact with respect to a knowledge graph. The generated result file can be uploaded to GERBIL for ROC AUC evaluation.

The methodology, dataset analysis, and experimental results are documented in the [project report](PROJECT_REPORT.md).

## GERBIL Result

The default offline model achieved a **ROC AUC of 0.6146 (61.46%)** on the
`SW 2022 Test` reference dataset, exceeding the course requirement of 60%.

- Experiment: [GERBIL 202607220010](https://gerbil-kbc.aksw.org/gerbil/experiment?id=202607220010)
- Task: `Fact Checking`
- Dataset: `SW 2022 Test`
- ROC AUC: `0.6146`

![GERBIL experiment result](docs/gerbil-result.png)

## Contributors

- Bofeng Duan
- Mukund Chavda

## Project Structure

```text
.
├── train.txt
├── test.txt
├── pyproject.toml
├── README.md
├── PROJECT_REPORT.md
├── docs/gerbil-result.png
├── src/kg_factcheck/
│   ├── cli.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── output.py
│   └── sparql.py
└── tests/
```

## Installation

Python 3.10 or newer is required. The core project uses only the Python standard library.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The included test suite uses `unittest`, so no additional test dependency is
required.

If package installation is not possible, run the CLI directly from the source tree:

```bash
PYTHONPATH=src python -m kg_factcheck.cli train --train train.txt --model model.joblib
PYTHONPATH=src python -m kg_factcheck.cli predict --test test.txt --model model.joblib --output result.ttl
PYTHONPATH=src python -m kg_factcheck.cli validate --result result.ttl --test test.txt
```

## Reproduce the Result File

Train the model:

```bash
kg-factcheck train --train train.txt --model model.joblib
```

Generate predictions:

```bash
kg-factcheck predict --test test.txt --model model.joblib --output result.ttl
```

Validate the GERBIL format:

```bash
kg-factcheck validate --result result.ttl --test test.txt
```

The output format is exactly one line per test fact:

```text
<Fact-URI> <http://swc2017.aksw.org/hasTruthValue> "0.8901000000"^^<http://www.w3.org/2001/XMLSchema#double> .
```

## Analyze and Benchmark Locally

Run a data summary and 5-fold ROC AUC estimate on the training data:

```bash
kg-factcheck analyze --train train.txt --test test.txt --folds 5
```

This local score is only a sanity check. The final grade-relevant result is the GERBIL evaluation.

## Model

The default model is a dependency-free hashed logistic regression blended with smoothed predicate priors. The default configuration intentionally trains for one epoch: on this small dataset, longer training overfits and gives worse cross-validation AUC. It uses:

- predicate identity and predicate tokens
- subject/object URI tokens
- subject/object seen-before indicators
- smoothed predicate, subject, and object truth rates
- simple lexical overlap and length features

This keeps the project executable on a clean machine while still going beyond a trivial majority baseline.

## Optional DBpedia Evidence

The default workflow is offline and reproducible. If network access is available, exact DBpedia triple evidence can be blended into predictions:

```bash
kg-factcheck predict \
  --test test.txt \
  --model model.joblib \
  --output result.ttl \
  --use-sparql \
  --sparql-weight 0.15
```

SPARQL responses are cached in `.kg_cache/dbpedia_exact.json`.

## GERBIL Upload

1. Go to `http://gerbil-kbc.aksw.org/gerbil/config`.
2. Choose `Fact Checking` as experiment type.
3. Upload `result.ttl`.
4. Choose `SW 2022 Test` as the reference dataset.
5. Submit and check the ROC AUC leaderboard result.

If GERBIL reports that the annotator could not be loaded, run:

```bash
kg-factcheck validate --result result.ttl --test test.txt
```

## Tests

```bash
python -m unittest discover -s tests
```

The current test suite contains four tests covering data parsing, GERBIL output,
model serialization and prediction, and ROC AUC calculation.

## AI Assistance

AI tools were used for initial repository organization, implementation
scaffolding, the report outline, and language editing. The team reviewed and
adapted these suggestions, ran the experiments and tests, checked the reported
values against the generated files and GERBIL result, and takes responsibility
for the submitted code and conclusions.
