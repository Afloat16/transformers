import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from xml.etree import ElementTree

HEAD = "d6a9829826927f13543ccc3e93a74ba2cde0aae3"
ROOT = Path.cwd()
TESTS = "tests/trainer/test_trainer_data.py"
WORKER = "tests/trainer/distributed/scripts/check_length_grouped_small_dataset.py"
SOURCE = "src/transformers/trainer_pt_utils.py"
PATHS = (SOURCE, TESTS, WORKER)
EVIDENCE = ROOT.parent / "evidence"
EVIDENCE.mkdir(exist_ok=True)
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == HEAD
for path in PATHS:
    assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path

import torch
import transformers
print("Candidate:", HEAD, flush=True)
print("Versions:", torch.__version__, transformers.__version__, flush=True)
env = dict(os.environ, OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
native = subprocess.run(
    [sys.executable, "-m", "pytest", TESTS + "::TrainerSmallLengthGroupedSamplerTest",
     TESTS + "::TrainerSamplerTest", "-q", "-p", "xdist.plugin", "-p", "pytest_env.plugin",
     f"--junitxml={EVIDENCE / 'native.xml'}"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, timeout=300,
)
(EVIDENCE / "native.log").write_text(native.stdout)
print(native.stdout, flush=True)
suite = ElementTree.parse(EVIDENCE / "native.xml").getroot()
totals = {k: sum(int(x.get(k, 0)) for x in suite.iter("testsuite"))
          for k in ("tests", "failures", "errors", "skipped")}
totals["returncode"] = native.returncode
assert totals == {"tests": 56, "failures": 0, "errors": 0, "skipped": 0, "returncode": 0}, totals

bandit = subprocess.run(
    [sys.executable, "-m", "bandit", "--format", "json", "-l", "--ignore-nosec",
     "-o", str(EVIDENCE / "candidate-bandit.json"), "--", *PATHS],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60,
)
report = json.loads((EVIDENCE / "candidate-bandit.json").read_text())
blocked = re.compile(r"^(B60[1-79]|B102|B307|B310|B323)$")
findings = [r for r in report["results"]
            if r["issue_severity"] == "HIGH" or blocked.match(r["test_id"])]
assert report["errors"] == [] and not findings, findings
print("Bandit 1.9.4: zero gate findings", flush=True)

worker = ROOT / WORKER
original = worker.read_text()
probes = []
anchors = {
    "initialization": "        torch.set_num_threads(1)",
    "training": "        torch.manual_seed(17)",
    "cleanup": "                dist.destroy_process_group()",
}
try:
    for stage, anchor in anchors.items():
        assert original.count(anchor) == 1
        indent = anchor[: len(anchor) - len(anchor.lstrip())]
        predicate = 'os.environ["RANK"] == "0"' if stage == "initialization" else "dist.get_rank() == 0"
        injection = (indent + "if " + predicate + ":\n" +
                     indent + "    print('watchdog probe reached " + stage + "', flush=True)\n" +
                     indent + "    time.sleep(30)\n" + anchor)
        mutated = original.replace("import copy\n", "import copy\nimport os\nimport time\n")
        mutated = mutated.replace("faulthandler.dump_traceback_later(180, exit=True)",
                                  "faulthandler.dump_traceback_later(5, exit=True)")
        mutated = mutated.replace(anchor, injection)
        worker.write_text(mutated)
        with tempfile.TemporaryDirectory() as output_dir:
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nnodes=1", "--nproc-per-node=3", str(worker), output_dir]
            started = time.monotonic()
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, env=env, timeout=90)
            elapsed = time.monotonic() - started
        (EVIDENCE / f"watchdog-{stage}.log").write_text(result.stdout)
        print(f"Watchdog {stage}: exit={result.returncode}, elapsed={elapsed:.2f}s", flush=True)
        print(result.stdout, flush=True)
        assert result.returncode != 0, stage
        assert f"watchdog probe reached {stage}" in result.stdout, stage
        assert "Timeout (0:00:05)" in result.stdout, stage
        assert elapsed < 60, (stage, elapsed)
        probes.append({"stage": stage, "returncode": result.returncode, "elapsed_seconds": elapsed})
        worker.write_text(original)
finally:
    worker.write_text(original)

for path in PATHS:
    assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"]), path
summary = {"head": HEAD, "torch": torch.__version__, "transformers": transformers.__version__,
           "native": totals, "bandit_gate_findings": findings, "watchdog_probes": probes,
           "deadline_scope": "180 seconds per worker main including initialization, training and cleanup; excludes interpreter imports and launcher startup"}
(EVIDENCE / "validation-result.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
