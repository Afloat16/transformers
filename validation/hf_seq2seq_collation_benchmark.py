"""Measure the native pretokenized T5 data-loading/collation path, not model compute."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader

import transformers
from transformers import BertTokenizer, DataCollatorForSeq2Seq, T5Config, T5ForConditionalGeneration


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()
torch.set_num_threads(2)
torch.manual_seed(0)
tokenizer = BertTokenizer(vocab={"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4})
config = T5Config(decoder_start_token_id=0)
model = T5ForConditionalGeneration(config)
collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
results = []
for batch_size, source_length, target_length in [(32, 128, 128), (32, 256, 512), (8, 512, 2048)]:
    size = batch_size * 48
    data = datasets.Dataset.from_dict(
        {
            "input_ids": [
                ((np.arange(source_length - i % 17) + i) % 30000 + 5).tolist() for i in range(size)
            ],
            "labels": [
                ((np.arange(target_length - i % 19) + 2 * i) % 30000 + 5).tolist() for i in range(size)
            ],
        }
    )
    for representation in ("numpy", "torch", None):
        formatted = data.with_format(representation)
        loader = DataLoader(formatted, batch_size=batch_size, num_workers=0, collate_fn=collator)
        durations = []
        signatures = []
        for repeat in range(6):
            begin = time.perf_counter()
            for batch in loader:
                pass
            elapsed = time.perf_counter() - begin
            if repeat:
                durations.append(elapsed / len(loader) * 1000)
            signatures.append(
                hashlib.sha256(
                    b"".join(batch[key].numpy().tobytes() for key in sorted(batch))
                ).hexdigest()
            )
        assert len(set(signatures)) == 1
        results.append(
            dict(
                batch_size=batch_size,
                source_length=source_length,
                target_length=target_length,
                representation=representation or "python",
                ms_per_batch=durations,
                median_ms=statistics.median(durations),
                last_batch_sha256=signatures[0],
            )
        )
        print(json.dumps(results[-1]), flush=True)
cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor())
report = dict(
    label=args.label,
    python=platform.python_version(),
    torch=torch.__version__,
    numpy=np.__version__,
    datasets=datasets.__version__,
    transformers=transformers.__version__,
    cpu=cpu,
    torch_threads=torch.get_num_threads(),
    model_config=config.to_dict(),
    parameters=sum(parameter.numel() for parameter in model.parameters()),
    batches_per_repeat=48,
    warmup_repeats=1,
    measured_repeats=5,
    scope="DataLoader iteration, Arrow formatting, dynamic padding and native decoder-input preparation; excludes model forward/backward",
    results=results,
)
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
