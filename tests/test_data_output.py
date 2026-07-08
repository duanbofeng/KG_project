from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_factcheck.data import parse_facts
from kg_factcheck.output import validate_result, write_result


class ParserAndOutputTests(unittest.TestCase):
    def test_parse_project_files(self):
        train = parse_facts(ROOT / "train.txt", require_truth=True)
        test = parse_facts(ROOT / "test.txt", require_truth=False)

        self.assertEqual(len(train), 1234)
        self.assertEqual(len(test), 1342)
        self.assertTrue(all(fact.truth in {0.0, 1.0} for fact in train))
        self.assertTrue(all(fact.truth is None for fact in test))
        self.assertEqual(len({fact.predicate for fact in train}), 9)
        self.assertEqual(len({fact.predicate for fact in test}), 9)

    def test_write_and_validate_result(self):
        test = parse_facts(ROOT / "test.txt", require_truth=False)[:3]
        scores = [0.0, 0.5, 1.0]

        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.ttl"
            write_result(result_path, test, scores)
            ok, errors = validate_result(result_path, test)

        self.assertTrue(ok, errors)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
