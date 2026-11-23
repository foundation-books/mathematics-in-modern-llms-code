"""Behavioral checks for companion coverage and process isolation."""

from contextlib import redirect_stderr, redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


spec = spec_from_file_location("companion_check", Path(__file__).resolve().parents[1] / "check.py")
checker = module_from_spec(spec)
spec.loader.exec_module(checker)


class CompanionChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "code"
        self.root.mkdir()
        self.chapter = dict(number=1, title="Fixture", directory="chapter_fixture",
                            starter="student.py", references=["reference.py"], tests=None)
        folder = self.root / self.chapter["directory"]
        folder.mkdir()
        for name in ("README.md", "requirements.txt", "student.py", "reference.py"):
            (folder / name).write_text("")
        self.write_manifest()

    def write_manifest(self, chapters=None):
        (self.root / "chapters.json").write_text(json.dumps(
            dict(schema_version=1, chapters=chapters or [self.chapter])))

    def test_current_release_covers_twenty_chapters_and_appendix_a(self):
        chapters = checker.load_chapters()
        self.assertEqual([chapter["number"] for chapter in chapters], list(range(1, 21)))
        self.assertEqual(
            [chapter["directory"] for chapter in chapters[3:6]],
            ["chapter_softmax_attention", "chapter_efficient_execution", "chapter_attention_state"],
        )
        self.assertEqual(
            [chapter["directory"] for chapter in chapters[-3:]],
            ["chapter_reinforcement_learning", "chapter_speculative_decoding", "chapter_discrete_diffusion"],
        )
        self.assertEqual([a["number"] for a in checker.load_appendices()], ["A"])

    def test_unlisted_appendix_rejects_partial_manifest(self):
        (self.root / "appendix_extra").mkdir()
        with self.assertRaisesRegex(ValueError, "Appendix directories.*manifest disagree"):
            checker.load_appendices(self.root)

    def test_appendix_only_selection_runs_no_chapter_programs(self):
        appendix = dict(self.chapter, number="A", title="Evaluation", directory="appendix_fixture")
        folder = self.root / appendix["directory"]
        folder.mkdir()
        for name in ("README.md", "requirements.txt", "student.py", "reference.py"):
            (folder / name).write_text("")
        manifest = self.root / "chapters.json"
        data = json.loads(manifest.read_text())
        data["appendices"] = [appendix]
        manifest.write_text(json.dumps(data))
        report = self.root.parent / "appendix-report.json"
        with redirect_stdout(io.StringIO()):
            status = checker.main(["--appendix", "A", "--report", str(report)], self.root)
        self.assertEqual(status, 0)
        data = json.loads(report.read_text())
        self.assertEqual(data["chapters"], [])
        self.assertEqual(data["appendices"], ["A"])
        self.assertEqual([r["chapter"] for r in data["results"]], ["A", "A"])

    def test_missing_reference_rejects_incomplete_bundle(self):
        (self.root / "chapter_fixture/reference.py").unlink()
        with self.assertRaisesRegex(ValueError, "Missing companion file"):
            checker.load_chapters(self.root)

    def test_unlisted_chapter_rejects_partial_manifest(self):
        (self.root / "chapter_extra").mkdir()
        with self.assertRaisesRegex(ValueError, "manifest disagree"):
            checker.load_chapters(self.root)

    def test_active_manuscript_drift_is_detected_when_sources_exist(self):
        book = self.root.parent / "book"
        book.mkdir()
        (book / "chapters.tex").write_text("\\input{chapter_missing/main}\n")
        with self.assertRaisesRegex(ValueError, "active book/chapters.tex"):
            checker.load_chapters(self.root)

    def test_duplicate_chapter_numbers_are_rejected(self):
        self.write_manifest([self.chapter, self.chapter])
        with self.assertRaisesRegex(ValueError, "Chapter numbers"):
            checker.load_chapters(self.root)

    def test_optional_notebook_matches_script_and_detects_code_drift(self):
        notebook = self.root / "student.ipynb"
        script = self.root / "student.py"
        notebook.write_text(json.dumps(dict(cells=[
            dict(cell_type="markdown", source=["# Instructions"]),
            dict(cell_type="code", source=["x = 1\n", "assert x == 1\n"])])))
        script.write_text("# Instructions\nx = 1\nassert x == 1\n")
        checker.check_notebook(notebook, script)
        script.write_text("x = 2\nassert x == 1\n")
        with self.assertRaisesRegex(ValueError, "code differ"):
            checker.check_notebook(notebook, script)

    def test_chapter_tests_do_not_share_generic_reference_imports(self):
        for number, name in enumerate(("first", "second"), 1):
            folder = self.root / name
            folder.mkdir()
            (folder / "reference.py").write_text(f"CHAPTER = {name!r}\n")
            (folder / "test_reference.py").write_text(
                "import unittest\nimport reference\n"
                "class LocalReference(unittest.TestCase):\n"
                f"    def test_identity(self):\n        self.assertEqual(reference.CHAPTER, {name!r})\n")
            chapter = dict(self.chapter, number=number, directory=name, tests="unittest")
            kind, command = list(checker.commands(chapter, self.root))[-1]
            self.assertEqual(kind, "tests")
            result = checker.run_command(command, self.root.parent, 10)
            self.assertEqual(result["exit_code"], 0, result["stderr"])
            self.assertIn("Ran 1 test", result["stderr"])

    def test_failed_program_sets_exit_status_and_retains_report(self):
        (self.root / "chapter_fixture/reference.py").write_text(
            "print('failure witness')\nraise SystemExit(3)\n")
        report = self.root.parent / "report.json"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = checker.main(["--chapter", "1", "--report", str(report)], self.root)
        self.assertEqual(status, 1)
        results = json.loads(report.read_text())["results"]
        self.assertEqual([result["exit_code"] for result in results], [0, 3])
        self.assertIn("failure witness", results[-1]["stdout"])

    def test_timeout_is_an_explicit_failure(self):
        command = [sys.executable, "-c", "import time; time.sleep(5)"]
        result = checker.run_command(command, self.root.parent, 0.05)
        self.assertEqual(result["exit_code"], 124)
        self.assertIn("Timed out", result["stderr"])

    def test_unknown_selection_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            checker.main(["--chapter", "26", "--list"], self.root)
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
