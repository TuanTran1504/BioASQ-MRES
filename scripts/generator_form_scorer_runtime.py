"""Freeze the audited evaluator distribution, including its manifest libraries."""
from pathlib import Path
import shutil
import tempfile
import zipfile

from scripts.prepare_answer_selector_data import read, digest, verify_files


def dependency_paths(jar):
    with zipfile.ZipFile(jar) as archive:
        manifest = archive.read("META-INF/MANIFEST.MF").decode("utf-8")
    # JAR manifests wrap attribute values onto lines beginning with one space.
    manifest = manifest.replace("\r\n", "\n").replace("\n ", "")
    names = next((line.split(":",1)[1].strip().split() for line in manifest.splitlines()
                  if line.lower().startswith("class-path:")), [])
    for name in names:
        path = (jar.parent / name).resolve()
        if not path.is_relative_to(jar.parent.resolve()) or not path.is_file():
            raise ValueError("Evaluator manifest dependency missing or outside its distribution: " + name)
    return names


def source(bundle):
    audit = read(bundle / "selector_audit_manifest.json")
    original = Path(audit["submission"])
    if digest(original / "prepared.json") != audit["prepared_sha256"]:
        raise ValueError("Original selector scorer preparation changed")
    state = read(original / "prepared.json")
    hashes = {n:v for n,v in state["files_sha256"].items() if n.startswith("scorer/")}
    if "scorer/BioASQEvaluation.jar" not in hashes:
        raise ValueError("No pinned official evaluator")
    verify_files(original, hashes)
    runtime = original / "scorer"
    if {p.relative_to(original).as_posix() for p in runtime.rglob("*") if p.is_file()} != set(hashes):
        raise ValueError("Official scorer distribution file membership changed")
    jar = runtime / "BioASQEvaluation.jar"
    for name in dependency_paths(jar):
        if "scorer/" + name not in hashes:
            raise ValueError("Unpinned evaluator dependency: " + name)
    return jar


def freeze(jar, destination):
    dependency_paths(jar)
    shutil.copytree(jar.parent, destination / "scorer")
    return {p.relative_to(destination).as_posix():digest(p)
            for p in (destination / "scorer").rglob("*") if p.is_file()}


def smoke(jar):
    """Exercise actual JSON parsing and matching before any expensive jobs."""
    from src.notebook_workflows.coverage_comparison import official_candidate_matches
    with tempfile.TemporaryDirectory(prefix="bioasq-scorer-smoke-") as directory:
        result = official_candidate_matches(
            [{"question_id":"runtime-probe", "question":"Runtime probe", "snippets":[], "gold_aliases":["probe-accepted"]}],
            [{"question_id":"runtime-probe","answer":a} for a in ("probe-accepted","probe-rejected")],
            Path(directory), jar_path=jar)
    if result != {("runtime-probe","probe-accepted"):True,("runtime-probe","probe-rejected"):False}:
        raise ValueError("Official scorer runtime smoke returned unexpected matching labels")
