import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest

from PIL import Image

from inference import get_class_names, init_db, run_full_pipeline, search_images


class TestInference(unittest.TestCase):
    def test_get_class_names(self):
        class_names = get_class_names()
        self.assertIsInstance(class_names, list)
        self.assertGreater(len(class_names), 0, "get_class_names should return a non-empty list")

    def test_init_db(self):
        try:
            init_db()
        except Exception as e:
            self.fail(f"init_db raised an exception: {e}")
        # Optionally, if init_db is expected to create folders/files,
        # add additional assertions here.
        # For example:
        # self.assertTrue(os.path.isdir("some_expected_directory"))

    def test_run_full_pipeline(self):
        # Create a dummy image (100x100 solid white)
        img = Image.new("RGB", (100, 100), color="white")
        result = run_full_pipeline(img)
        self.assertIsInstance(result, dict)
        self.assertIn("bathroom_type", result, "Result should contain 'bathroom_type'")
        self.assertIn("detected_objects", result, "Result should contain 'detected_objects'")
        self.assertIsInstance(result["detected_objects"], list, "'detected_objects' should be a list")

    def test_search_images(self):
        # Test with a query that is unlikely to return any matches
        matches = search_images("nonexistent_query")
        self.assertIsInstance(matches, list)
        # Depending on implementation, you might expect an empty list
        # or handle a mocked environment. This test ensures the output is a list.
        # For example, if there is setup for images, consider adding more detailed tests.

if __name__ == "__main__":
    unittest.main()
