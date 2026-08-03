import unittest

from retrieval._6_hyde import HyDE


class HyDEModelResolutionTest(unittest.TestCase):
    def test_hyde_strips_free_suffix_from_model_name(self):
        hyde = HyDE()

        model = hyde._resolve_model("meta-llama/llama-3.2-3b-instruct:free")

        self.assertEqual(model, "meta-llama/llama-3.2-3b-instruct")


if __name__ == "__main__":
    unittest.main()
