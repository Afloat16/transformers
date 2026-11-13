import tempfile
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

torch.set_num_threads(2)
torch.manual_seed(7)
model = GPT2LMHeadModel(GPT2Config())
rows = [{"input_ids": torch.tensor([1, 7, 3, 9, 4, 8, 6, 2]) + i} for i in range(3)]
for row in rows:
    row["labels"] = row["input_ids"].clone()
with tempfile.TemporaryDirectory() as output_dir:
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir, use_cpu=True, per_device_eval_batch_size=2,
            eval_accumulation_steps=1, report_to=[], disable_tqdm=True,
        ),
        eval_dataset=rows,
        preprocess_logits_for_metrics=lambda logits, labels: logits.to(torch.float8_e4m3fn),
        compute_metrics=lambda output: {"logit_mean": output.predictions.mean().item()},
    )
    result = trainer.predict(rows)
    model.eval()
    expected = []
    with torch.no_grad():
        for start in range(0, len(rows), 2):
            inputs = torch.stack([row["input_ids"] for row in rows[start:start + 2]])
            expected.append(model(input_ids=inputs).logits.to(torch.float8_e4m3fn).float().numpy())
    expected = np.concatenate(expected)
    np.testing.assert_array_equal(result.predictions, expected)
    assert result.predictions.dtype == np.float32
    assert result.metrics["test_logit_mean"] == expected.mean().item()
    np.testing.assert_array_equal(result.label_ids, np.stack([row["labels"].numpy() for row in rows]))
    print("Default GPT2Config Trainer.predict preserved all 3 x 8 x 50257 quantized logits and metric.", flush=True)
