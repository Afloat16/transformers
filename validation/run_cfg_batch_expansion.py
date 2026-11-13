import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "acf8004ec93d3d3195ea4945c693ff1c8959374e"
HEAD = "2ae942fc719794c5ea5ce8c1482d8f5defff9635"
SOURCE = "src/transformers/generation/utils.py"
TESTS = "tests/generation/test_cfg_batch_expansion.py"
subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", BASE, HEAD], check=True)
for path in (SOURCE, TESTS):
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
import torch
import transformers
print("Versions:", torch.__version__, transformers.__version__, flush=True)
def run(label, paths, expression=None):
    command = [sys.executable, "-m", "pytest", *paths, "-q", "-p", "xdist.plugin", "-p", "pytest_env.plugin", f"--junitxml={label}.xml"]
    if expression:
        command += ["-k", expression]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    Path(f"{label}.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    root = ElementTree.parse(f"{label}.xml").getroot()
    totals = {k: sum(int(x.get(k, 0)) for x in root.iter("testsuite")) for k in ("tests", "failures", "errors", "skipped")}
    totals["returncode"] = result.returncode
    print(label, totals, flush=True)
    return totals

fixed = Path(SOURCE).read_bytes()
try:
    Path(SOURCE).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"]))
    before = run("before", [TESTS])
finally:
    Path(SOURCE).write_bytes(fixed)
assert before["tests"] == 36 and before["failures"] == 18, before
assert before["errors"] == before["skipped"] == 0 and before["returncode"] == 1, before
after = run("after", [TESTS])
assert after == {"tests": 36, "failures": 0, "errors": 0, "skipped": 0, "returncode": 0}, after
existing = run("existing", ["tests/generation/test_logits_process.py"], "classifier_free_guidance")
assert existing["tests"] > 0 and existing["returncode"] == 0, existing
for path in (SOURCE, TESTS):
    assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
result = {"head": HEAD, "base": BASE, "torch": torch.__version__, "before": before, "after": after, "existing": existing}
Path("validation_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
