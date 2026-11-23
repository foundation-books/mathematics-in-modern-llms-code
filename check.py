"""Run chapter and appendix starters, references and tests in fresh processes."""

from __future__ import annotations

import argparse
import ast
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


CODE_ROOT = Path(__file__).resolve().parent


def local_file(folder, name, suffix):
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(suffix):
        raise ValueError(f"Expected a local {suffix} filename in {folder}: {name!r}")
    file = folder / name
    if not file.is_file():
        raise ValueError(f"Missing companion file: {file}")
    return file


def check_notebook(notebook, script):
    """The optional notebook and its script must contain the same Python code."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    source = "\n\n".join("".join(cell["source"]) for cell in cells
                         if cell["cell_type"] == "code")
    notebook_code = ast.dump(ast.parse(source), include_attributes=False)
    script_code = ast.dump(ast.parse(script.read_text(encoding="utf-8")), include_attributes=False)
    if notebook_code != script_code:
        raise ValueError(f"Notebook and script code differ: {notebook}, {script}")


def load_chapters(code_root=CODE_ROOT):
    data = json.loads((code_root / "chapters.json").read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported companion manifest version")
    chapters = data.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("The companion manifest must list its chapters")
    directories = []
    for number, chapter in enumerate(chapters, 1):
        if type(chapter.get("number")) is not int or chapter["number"] != number:
            raise ValueError("Chapter numbers must appear once in consecutive reading order")
        directory = chapter.get("directory", "")
        if not isinstance(directory, str) or not re.fullmatch(r"chapter_[a-z_]+", directory):
            raise ValueError(f"Invalid chapter directory: {directory!r}")
        if not isinstance(chapter.get("title"), str) or not chapter["title"].strip():
            raise ValueError(f"Chapter {number} needs a title")
        directories.append(directory)
        folder = code_root / directory
        local_file(folder, "README.md", ".md")
        local_file(folder, "requirements.txt", ".txt")
        starter = local_file(folder, chapter["starter"], ".py")
        references = chapter.get("references")
        if not isinstance(references, list) or not references or len(references) != len(set(references)):
            raise ValueError(f"Chapter {number} needs distinct executable references")
        for reference in references:
            local_file(folder, reference, ".py")
        runner = chapter.get("tests")
        if runner not in (None, "unittest", "pytest"):
            raise ValueError(f"Unsupported test runner: {runner!r}")
        tests = list(folder.glob("test_*.py"))
        if bool(tests) != (runner is not None):
            raise ValueError(f"Test files and manifest disagree for chapter {number}")
        if "notebook" in chapter:
            notebook = local_file(folder, chapter["notebook"], ".ipynb")
            check_notebook(notebook, starter)
    present = {folder.name for folder in code_root.glob("chapter_*") if folder.is_dir()}
    if len(directories) != len(set(directories)) or present != set(directories):
        raise ValueError("Chapter directories and the companion manifest disagree")
    manuscript = code_root.parent / "book/chapters.tex"
    if manuscript.is_file():
        source = re.sub(r"(?m)%.*$", "", manuscript.read_text(encoding="utf-8"))
        implemented = re.findall(r"\\input\{(chapter_[^/}]+)/main\}", source)
        if implemented != directories:
            raise ValueError("Companion coverage differs from active book/chapters.tex inputs")
    return chapters


def commands(chapter, code_root=CODE_ROOT):
    folder = code_root / chapter["directory"]
    yield "starter", [sys.executable, str(folder / chapter["starter"])]
    for reference in chapter["references"]:
        yield "reference", [sys.executable, str(folder / reference)]
    if chapter["tests"] == "unittest":
        yield "tests", [sys.executable, "-m", "unittest", "discover",
                        "-s", str(folder), "-p", "test_*.py"]
    elif chapter["tests"] == "pytest":
        yield "tests", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(folder)]


def load_appendices(code_root=CODE_ROOT):
    data = json.loads((code_root / "chapters.json").read_text(encoding="utf-8"))
    appendices = data.get("appendices", [])
    if not isinstance(appendices, list):
        raise ValueError("The appendix inventory must be a list")
    identifiers, directories = [], []
    for appendix in appendices:
        identifier = appendix.get("number", "")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Z]", identifier):
            raise ValueError("Appendix identifiers must be single capital letters")
        directory = appendix.get("directory", "")
        if not isinstance(directory, str) or not re.fullmatch(r"appendix_[a-z_]+", directory):
            raise ValueError(f"Invalid appendix directory: {directory!r}")
        if not isinstance(appendix.get("title"), str) or not appendix["title"].strip():
            raise ValueError(f"Appendix {identifier} needs a title")
        identifiers.append(identifier)
        directories.append(directory)
        folder = code_root / directory
        local_file(folder, "README.md", ".md")
        local_file(folder, "requirements.txt", ".txt")
        local_file(folder, appendix["starter"], ".py")
        references = appendix.get("references")
        if not isinstance(references, list) or not references or len(references) != len(set(references)):
            raise ValueError(f"Appendix {identifier} needs distinct executable references")
        for reference in references:
            local_file(folder, reference, ".py")
        runner = appendix.get("tests")
        if runner not in (None, "unittest", "pytest"):
            raise ValueError(f"Unsupported test runner: {runner!r}")
        if bool(list(folder.glob("test_*.py"))) != (runner is not None):
            raise ValueError(f"Test files and manifest disagree for appendix {identifier}")
    present = {folder.name for folder in code_root.glob("appendix_*") if folder.is_dir()}
    if (len(identifiers) != len(set(identifiers)) or
            len(directories) != len(set(directories)) or present != set(directories)):
        raise ValueError("Appendix directories and the companion manifest disagree")
    manuscript = code_root.parent / "book/main.tex"
    if manuscript.is_file():
        source = re.sub(r"(?m)%.*$", "", manuscript.read_text(encoding="utf-8"))
        for directory in directories:
            if rf"\input{{{directory}}}" not in source:
                raise ValueError("Companion appendix differs from active book/main.tex inputs")
    return appendices


def run_command(command, cwd, timeout):
    started = time.monotonic()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    try:
        result = subprocess.run(command, cwd=cwd, env=environment, text=True,
                                capture_output=True, timeout=timeout, check=False)
        record = dict(command=command, exit_code=result.returncode,
                      stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired as error:
        def decode(output):
            return output.decode(errors="replace") if isinstance(output, bytes) else output or ""
        record = dict(command=command, exit_code=124, stdout=decode(error.stdout),
                      stderr=decode(error.stderr) + f"\nTimed out after {timeout} seconds")
    record["seconds"] = round(time.monotonic() - started, 3)
    return record


def main(argv=None, code_root=CODE_ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", type=int, nargs="+", help="Check only these chapter numbers")
    parser.add_argument("--appendix", nargs="+", help="Check only these appendix letters (for example A)")
    parser.add_argument("--list", action="store_true", help="Validate and list chapter coverage")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds allowed per program (default: 600)")
    parser.add_argument("--report", type=Path, help="Save versions, commands and full outputs as JSON")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if sys.flags.optimize:
        parser.error("Run without -O or PYTHONOPTIMIZE; the mathematical checks use assertions")
    try:
        chapters = load_chapters(code_root)
        appendices = load_appendices(code_root)
    except (ValueError, KeyError, TypeError, OSError, SyntaxError) as error:
        print(f"Invalid companion: {error}", file=sys.stderr)
        return 1
    requested = set(args.chapter or ([] if args.appendix else [chapter["number"] for chapter in chapters]))
    unknown = requested - {chapter["number"] for chapter in chapters}
    if unknown:
        parser.error(f"Unknown chapter numbers: {sorted(unknown)}")
    selected = [chapter for chapter in chapters if chapter["number"] in requested]
    requested_appendices = set(args.appendix or ([] if args.chapter else [a["number"] for a in appendices]))
    unknown_appendices = requested_appendices - {a["number"] for a in appendices}
    if unknown_appendices:
        parser.error(f"Unknown appendix letters: {sorted(unknown_appendices)}")
    selected_appendices = [a for a in appendices if a["number"] in requested_appendices]
    if args.list:
        for chapter in selected:
            print(f"{chapter['number']:2}  {chapter['directory']}  {chapter['title']}")
        for appendix in selected_appendices:
            print(f" {appendix['number']}  {appendix['directory']}  {appendix['title']}")
        return 0
    packages = {}
    required = ["sympy", "mpmath"]
    if any(chapter["tests"] == "pytest" for chapter in selected + selected_appendices):
        required.append("pytest")
    for package in required:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            print(f"Missing {package}; install {code_root / 'requirements-dev.txt'}", file=sys.stderr)
            return 1
    print(f"Python {platform.python_version()}; " + "; ".join(f"{p} {v}" for p, v in packages.items()), flush=True)
    results = []

    def execute(number, kind, command):
        record = run_command(command, code_root.parent, args.timeout)
        record.update(chapter=number, kind=kind)
        results.append(record)
        status = "PASS" if record["exit_code"] == 0 else "FAIL"
        label = Path(command[1]).name if kind != "tests" else "tests"
        identifier = f"{number:02}" if isinstance(number, int) else number
        print(f"[{identifier}] {status} {kind}: {label} ({record['seconds']:.1f}s)", flush=True)
        if record["exit_code"]:
            print((record["stdout"] + record["stderr"])[-12000:], file=sys.stderr, flush=True)

    for chapter in selected + selected_appendices:
        for kind, command in commands(chapter, code_root):
            execute(chapter["number"], kind, command)
    if args.chapter is None and args.appendix is None:
        execute(0, "tests", [sys.executable, "-m", "unittest", "discover",
                             "-s", str(code_root / "tests"), "-p", "test_*.py"])
    failed = sum(record["exit_code"] != 0 for record in results)
    report = dict(python=platform.python_version(), packages=packages,
                  chapters=[chapter["number"] for chapter in selected],
                  appendices=[a["number"] for a in selected_appendices], results=results)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(selected)} chapters; {len(selected_appendices)} appendices; "
          f"{len(results)} programs; {failed} failures.", flush=True)
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
