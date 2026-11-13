import json
import tempfile
import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

torch.set_num_threads(2)
torch.manual_seed(7)
model = GPT2LMHeadModel(GPT2Config())
rows = [[1 + i, 7 + i, 3 + i, 9 + i, 4 + i, 8 + i, 6 + i, 2 + i] for i in range(7)]
dataset = Dataset.from_dict({"input_ids": rows, "labels": rows})
results = {}
with tempfile.TemporaryDirectory() as directory:
    for dispatch in (False, True):
        for streaming in (False, True):
            data = dataset.to_iterable_dataset() if streaming else dataset
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory, use_cpu=True, per_device_eval_batch_size=3,
                    report_to=[], disable_tqdm=True,
                    accelerator_config={"dispatch_batches": dispatch},
                ),
                eval_dataset=data,
            )
            result = trainer.evaluate()
            results[f"dispatch_{dispatch}_streaming_{streaming}"] = result["eval_loss"]
print("STREAMING_LOSS_PROBE " + json.dumps(results, sort_keys=True), flush=True)
delta = abs(results["dispatch_False_streaming_True"] - results["dispatch_False_streaming_False"])
assert delta > 1e-5, results
assert abs(results["dispatch_True_streaming_True"] - results["dispatch_True_streaming_False"]) < 1e-5, results
