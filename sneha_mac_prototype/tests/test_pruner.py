
import importlib
import unittest


class TestProjectModules(unittest.TestCase):
    def test_core_modules_import(self):
        module_names = [
            "src.model_loader",
            "src.entropy_calculator",
            "src.budget_allocator",
            "src.token_selector",
            "src.cache_pruner",
        ]

        for module_name in module_names:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
