import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "41f9d3992023970fb5a463e70d56232506c80fbc"
HEAD = "bf9a3cbe64372541b9be46b691f033301e6fc1b8"
SOURCE = "src/transformers/trainer_pt_utils.py"
TESTS = "tests/trainer/test_trainer_evaluation.py"
TARGET = TESTS + "::TrainerEvaluationTest::test_predict_preprocessed_float8_logits"
subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", BASE, HEAD], check=True)
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
        [sys.executable, "validation/probe_float8_predictions.py"],
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
assert "unsupported ScalarType Float8" in realistic_before.stdout, realistic_before.stdout
after = run("after", [TARGET])
assert after["tests"] > 0 and after["failures"] == after["errors"] == after["skipped"] == after["returncode"] == 0, after
realistic_after = probe("default_gpt2_after")
assert realistic_after.returncode == 0, realistic_after.stdout
existing = run("existing", [
    TESTS + "::TrainerEvaluationTest::test_evaluate",
    TESTS + "::TrainerEvaluationTest::test_predict",
    TESTS + "::TrainerBatchEvalMetricsTest::test_evaluate_with_batch_eval_metrics",
    TESTS + "::TrainerBatchEvalMetricsTest::test_predict_with_batch_eval_metrics",
])
assert existing["tests"] == 4 and existing["returncode"] == 0, existing
for path in (SOURCE, TESTS):
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
Path("validation_result.json").write_text(json.dumps({
    "head": HEAD, "base": BASE, "before": before, "after": after,
    "existing": existing, "default_gpt2_before": realistic_before.returncode,
    "default_gpt2_after": realistic_after.returncode,
}, indent=2) + "\n")
