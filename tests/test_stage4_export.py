import sqlite3
import unittest

from scripts.data_pipeline.export.run_stage4_export import (
    _db_percentiles,
    choose_split_counts,
    group_id,
    percentiles,
)


class Stage4Tests(unittest.TestCase):
    def test_question_group_uses_shared_normalization(self):
        self.assertEqual(group_id("  Article   32?\n"), group_id("Article 32?"))

    def test_group_assignment_is_deterministic_and_complete(self):
        sizes = {f"group-{i}": 1 for i in range(1000)}
        first = choose_split_counts(sizes, 20261031)
        self.assertEqual(first, choose_split_counts(sizes, 20261031))
        self.assertEqual(set(first), set(sizes))
        self.assertEqual(set(first.values()), {"train", "validation", "test"})

    def test_percentiles(self):
        self.assertEqual(percentiles([1, 2, 3, 4, 5]),
                         {"median": 3, "p90": 5, "p95": 5, "p99": 5, "max": 5})
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE token_lengths(split,n)")
        db.executemany("INSERT INTO token_lengths VALUES('train',?)", [(n,) for n in range(1, 6)])
        self.assertEqual(_db_percentiles(db, "train", 5), percentiles([1, 2, 3, 4, 5]))


if __name__ == "__main__":
    unittest.main()
