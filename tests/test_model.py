from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_factcheck.data import parse_facts
from kg_factcheck.model import FactCheckModel, TrainingConfig, roc_auc


class ModelTests(unittest.TestCase):
    def test_train_save_load_predict(self):
        facts = parse_facts(ROOT / "train.txt", require_truth=True)[:120]
        config = TrainingConfig(epochs=3, num_buckets=512)
        model = FactCheckModel.train(facts, config=config)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            model.save(path)
            loaded = FactCheckModel.load(path)

        scores = loaded.predict(facts[:10])
        self.assertEqual(len(scores), 10)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

    def test_roc_auc(self):
        labels = [0.0, 0.0, 1.0, 1.0]
        scores = [0.1, 0.4, 0.35, 0.8]
        self.assertAlmostEqual(roc_auc(labels, scores), 0.75)


if __name__ == "__main__":
    unittest.main()
