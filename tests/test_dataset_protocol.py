"""Checks for the manuscript's optional SegMine working-subset audit."""
import unittest

from minediffroad.data import SEGMINE_PAPER_SPLITS, verify_paper_segmine


def paper_rows():
    rows = []
    for split, counts in SEGMINE_PAPER_SPLITS.items():
        for index in range(counts["frames"]):
            scene_index = index % counts["scenes"]
            group_index = scene_index * counts["groups"] // counts["scenes"]
            rows.append({"id": f"{split}-{index}", "split": split,
                         "scene": f"{split}-scene-{scene_index}",
                         "group": f"{split}-group-{group_index}"})
    return rows


class DatasetProtocolTests(unittest.TestCase):
    def test_corrected_segmine_counts(self):
        result = verify_paper_segmine(paper_rows())
        self.assertEqual(result["records"], 749)
        self.assertEqual(result["splits"], SEGMINE_PAPER_SPLITS)

    def test_missing_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SegMine test counts"):
            verify_paper_segmine(paper_rows()[:-1])

    def test_cross_split_scene_is_rejected(self):
        rows = paper_rows()
        rows[-1]["scene"] = rows[0]["scene"]
        with self.assertRaisesRegex(ValueError, "scene .* crosses dataset splits"):
            verify_paper_segmine(rows)

    def test_missing_scene_is_rejected(self):
        rows = paper_rows()
        del rows[0]["scene"]
        with self.assertRaisesRegex(ValueError, "require nonempty scene"):
            verify_paper_segmine(rows)


if __name__ == "__main__":
    unittest.main()
