"""Why does RL get zero reward? Look at what the model actually writes.

Stop the v8 training cell first (it holds both GPUs), then run this in the same
session: it imports the task generator from /kaggle/working/train_v8_grpo.py.

For stages 1-4 it samples the model on a few training tasks, runs the code, and
prints plan, code, expected output and actual output (or error). Three things
to look for:
  - plan right, code arithmetic wrong   -> skill gap, fixable with basic SFT data
  - plan itself wrong or copied template -> model matches templates, not tasks
  - output looks right but graded wrong  -> bug in my grading
"""
import json
import random
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/kaggle/working")
sys.path.insert(0, "/kaggle/working/EsolangBench")
import train_v8_grpo as v8
from bf_gym.interpreter import run_bf

ADAPTER = "hd2514p/bf-qwen-1.5b-sft-v7-plan"   # also try v8.HUB_RL
TASKS_PER_STAGE = 3
SAMPLES = 3

tokenizer = AutoTokenizer.from_pretrained(v8.BASE_ID)
model = AutoModelForCausalLM.from_pretrained(v8.BASE_ID, dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
doc_text = open(f"{v8.REPO_DIR}/esolang_bench/docs/brainfuck.md", encoding="utf-8").read()

rng = random.Random(7)
summary = []
for stage in range(4):
    print(f"\n{'#' * 70}\nSTAGE {stage + 1}: {v8.STAGE_NAMES[stage]}\n{'#' * 70}")
    for _ in range(TASKS_PER_STAGE):
        task = v8.make_task(stage, rng, False, doc_text)
        tests = json.loads(task["tests"])
        print(f"\nTASK: {task['prompt'][-1]['content'][-160:]!r}")
        inputs = tokenizer.apply_chat_template(task["prompt"], add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=v8.MAX_COMPLETION, do_sample=True,
                                 temperature=1.0, num_return_sequences=SAMPLES,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        for j, seq in enumerate(out):
            raw = tokenizer.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            code = v8.extract_code(raw)
            inp, exp = tests[0]
            res = run_bf(code, stdin=inp.encode("latin-1"), config=v8.STRICT)
            score = v8.pass_fraction(code, task["tests"])
            summary.append((stage, score))
            print(f"\n  --- sample {j + 1}  pass {score:.2f}")
            print("  " + raw.strip().replace("\n", "\n  "))
            got = repr(res.output)
            got = got if len(got) <= 80 else got[:80] + f"... ({len(res.output)} bytes)"
            print(f"  input {inp!r}  expected {exp!r}  got {got}  error {res.error}")

print("\n\nmean pass per stage:")
for stage in range(4):
    s = [x for st, x in summary if st == stage]
    print(f"  stage {stage + 1} {v8.STAGE_NAMES[stage]:12} {sum(s) / len(s):.2f}")
