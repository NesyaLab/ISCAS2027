"""Check that a shared archive is complete but contains no local environment."""

from pathlib import Path
import tempfile
import unittest
import zipfile

from start import package_project


class PackagingTests(unittest.TestCase):
    def test_portable_archive_contents(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary) / "project"
            for filename in (
                "start.py", "requirements-lock.txt", "notebook.ipynb",
                "results/run/table.csv", ".venv/secret.txt", "__pycache__/module.pyc",
                ".venv-moved-123/old.txt", ".runtime/kernel.json",
            ):
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test", encoding="utf-8")
            archive = package_project(root)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()), {
                    "project/start.py", "project/requirements-lock.txt",
                    "project/notebook.ipynb", "project/results/run/table.csv",
                })
                self.assertIsNone(z.testzip())


if __name__ == "__main__":
    unittest.main()
