import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "acf8004ec93d3d3195ea4945c693ff1c8959374e"
HEAD = "70f2d58479c7f204648bfad1a67a08435d453e72"
SOURCE = "src/transformers/trainer_pt_utils.py"
TESTS = "tests/trainer/test_trainer_data.py"
ROOT = Path.cwd()
subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", BASE, HEAD], check=True)
for path in (SOURCE, TESTS):
    assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
import torch
import transformers
print("Versions:", torch.__version__, transformers.__version__, flush=True)
print("Production head:", HEAD, flush=True)
environment = dict(os.environ, OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
def run(label, expression=None):
    args = [sys.executable, "-m", "pytest", TESTS + "::TrainerLabelSmoothingTest", "-q",
            "-p", "xdist.plugin", "-p", "pytest_env.plugin", f"--junitxml={label}.xml"]
    if expression:
        args += ["-k", expression]
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=environment)
    Path(f"{label}.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    root = ElementTree.parse(f"{label}.xml").getroot()
    totals = {k: sum(int(x.get(k, 0)) for x in root.iter("testsuite"))
              for k in ("tests", "failures", "errors", "skipped")}
    totals["returncode"] = result.returncode
    print(label, totals, flush=True)
    return totals

fixed = (ROOT / SOURCE).read_bytes()
try:
    (ROOT / SOURCE).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"]))
    before = run("before", "low_precision_reduction or preserves_reduction_precision or half_model_updates")
finally:
    (ROOT / SOURCE).write_bytes(fixed)
assert before["tests"] == 30 and before["failures"] >= 12, before
assert before["errors"] == before["skipped"] == 0 and before["returncode"] == 1, before
after = run("after")
assert after == {"tests": 32, "failures": 0, "errors": 0, "skipped": 0, "returncode": 0}, after
for path in (SOURCE, TESTS):
    assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
summary = {"head": HEAD, "base": BASE, "torch": torch.__version__, "transformers": transformers.__version__,
           "before": before, "after": after}
Path("validation_result.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
