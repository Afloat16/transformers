import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "41f9d3992023970fb5a463e70d56232506c80fbc"
HEAD = "bbfc2fe3c40a2d017f52934a6cb0d43aeee4710c"
SOURCE = "src/transformers/trainer.py"
TESTS = "tests/trainer/test_trainer_data.py"
TARGET = TESTS + "::TrainerDynamicShapesAndIterableTest::test_iterable_evaluation_loss_uses_actual_batch_size"
subprocess.run(["git", "fetch", "--no-tags", "origin", BASE, HEAD], check=True)
for path in (SOURCE, TESTS):
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path

def run(label, tests):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *tests, "-q", "-p", "xdist.plugin",
         "-p", "pytest_env.plugin", f"--junitxml={label}.xml"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    Path(f"{label}.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    root = ElementTree.parse(f"{label}.xml").getroot()
    totals = {key: sum(int(s.get(key, 0)) for s in root.iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    totals["returncode"] = result.returncode
    print(label, totals, flush=True)
    return totals

def probe(label):
    result = subprocess.run(
        [sys.executable, "validation/probe_streaming_eval_loss.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    Path(label + ".log").write_text(result.stdout)
    print(result.stdout, flush=True)
    return result

fixed = Path(SOURCE).read_bytes()
try:
    Path(SOURCE).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"]))
    before = run("before", [TARGET])
    realistic_before = probe("default_gpt2_before")
finally:
    Path(SOURCE).write_bytes(fixed)
assert before["failures"] > 0 and before["errors"] == 0 and before["returncode"] == 1, before
assert realistic_before.returncode != 0, realistic_before.stdout
assert "STREAMING_LOSS_PROBE" in realistic_before.stdout and "AssertionError" in realistic_before.stdout, realistic_before.stdout
after = run("after", [TARGET])
assert after["tests"] > 0 and after["failures"] == after["errors"] == after["skipped"] == after["returncode"] == 0, after
realistic_after = probe("default_gpt2_after")
assert realistic_after.returncode == 0, realistic_after.stdout
existing = run("existing", [
    TESTS + "::TrainerDynamicShapesAndIterableTest::test_evaluation_iterable_dataset",
    TESTS + "::TrainerDynamicShapesAndIterableTest::test_predict_iterable_dataset",
    "tests/trainer/test_trainer_evaluation.py::TrainerEvaluationTest::test_evaluate",
    "tests/trainer/test_trainer_evaluation.py::TrainerEvaluationTest::test_predict",
])
assert existing["tests"] == 4 and existing["returncode"] == 0, existing
for path in (SOURCE, TESTS):
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
Path("validation_result.json").write_text(json.dumps({
    "head": HEAD, "base": BASE, "before": before, "after": after,
    "existing": existing, "default_gpt2_before": realistic_before.returncode,
    "default_gpt2_after": realistic_after.returncode,
}, indent=2) + "\n")
