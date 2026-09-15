"""Kaggle: OFFICIAL EsolangBench harness on v8 final - zero_shot, then self_scaffolding.

The harness runs unmodified except two patches: its OpenRouter URL points at our
local server, and a 200k-step guard stops non-terminating programs from hanging
the run (their interpreter has no step limit). Scoring, prompts, test cases and
the strict interpreter are theirs.

Our server returns only the code after CODE (the plan's . , - are Brainfuck
commands). It honours the temperature the harness sends, so the 5
self_scaffolding attempts actually differ - the earlier run hardcoded greedy
decoding and produced 5 identical attempts.

2/80 = 2.5%. Beating it needs 3+ problems.

Setup: GPU T4 (one is used), Internet on, secret HF_TOKEN.
"""
!pip install -q peft accelerate flask
!pip install -q -U transformers peft
!pip uninstall -y -q torchao

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import json
import logging
import re
import socket
import subprocess
import sys
import threading
import time

import requests
import torch
from flask import Flask, jsonify, request
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

login(token=UserSecretsClient().get_secret("HF_TOKEN"))

BASE_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER = "hd2514p/bf-qwen-1.5b-grpo-v8b"
MODEL_KEY = "bf-qwen-1.5b-grpo-v8b"
OUT_DIR = "/kaggle/working"
REPO_DIR = f"{OUT_DIR}/EsolangBench"
TEMPERATURE = 0.7      # used by self_scaffolding; zero_shot is greedy below
MAX_NEW = 768

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
model = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print("loaded", ADAPTER, flush=True)


def extract_code(text):
    return text.rsplit("\nCODE\n", 1)[-1].strip() if "\nCODE\n" in text else text.strip()


app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)   # no per-request log lines
state = {"temperature": 0.0, "calls": 0}


@app.route("/api/v1/chat/completions", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    temp = state["temperature"]
    inputs = tokenizer.apply_chat_template(body.get("messages", []), add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=temp > 0,
                             temperature=max(temp, 1e-5), top_p=0.95,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    state["calls"] += 1
    return jsonify({"choices": [{"message": {"role": "assistant", "content": extract_code(raw)}}]})


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    PORT = s.getsockname()[1]
URL = f"http://127.0.0.1:{PORT}/api/v1/chat/completions"
threading.Thread(target=lambda: app.run(host="127.0.0.1", port=PORT, threaded=False),
                 daemon=True).start()
time.sleep(2)
r = requests.post(URL, json={"messages": [{"role": "user", "content": "Print 'A' in Brainfuck."}]},
                  timeout=300)
print("server check:", r.json()["choices"][0]["message"]["content"][:60], flush=True)

# --- clone + patch harness (idempotent) ---
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/Lossfunk/EsolangBench.git", REPO_DIR], check=True)
cfg = f"{REPO_DIR}/esolang_bench/benchmarking/config.py"
src = open(cfg, encoding="utf-8").read()
src, n = re.subn(r'OPENROUTER_BASE_URL\s*=\s*".*?"', f'OPENROUTER_BASE_URL = "{URL}"', src)
assert n, "could not patch OPENROUTER_BASE_URL"
if f'"{MODEL_KEY}"' not in src:
    src = src.replace("MODEL_NAME_TO_ID = {", f'MODEL_NAME_TO_ID = {{\n    "{MODEL_KEY}": "{MODEL_KEY}",')
assert f'"{MODEL_KEY}"' in src, "could not register model"
open(cfg, "w", encoding="utf-8").write(src)

bf = f"{REPO_DIR}/esolang_bench/interpreters/brainfuck.py"
bsrc = open(bf, encoding="utf-8").read()
anchor = "            if len(last_ops) > 12:\n                last_ops.pop(0)\n"
guard = ("            if steps > 200_000:\n"
         "                return ExecutionResult(\n"
         "                    stdout=\"\".join(stdout_chars),\n"
         "                    stderr=\"Brainfuck runtime error: step limit exceeded\",\n"
         "                    exit_code=1,\n"
         "                    error_type=\"timeout\",\n"
         "                    trace=self._trace(steps, pointer, tape, last_ops),\n"
         "                )\n")
if guard not in bsrc:
    assert anchor in bsrc, "could not patch step limit"
    open(bf, "w", encoding="utf-8").write(bsrc.replace(anchor, anchor + guard))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO_DIR}[benchmark]"], check=True)

data_dir = f"{REPO_DIR}/esolang_bench/data"
merged = []
for tier in ("easy", "medium", "hard", "extra_hard"):
    merged.extend(json.load(open(f"{data_dir}/esolang_{tier}_dataset.json", encoding="utf-8"))["problems"])
assert len(merged) == 80
merged_path = f"{OUT_DIR}/esolang_all.json"
json.dump({"problems": merged}, open(merged_path, "w", encoding="utf-8"))


def run(regime, temperature):
    state["temperature"], state["calls"] = temperature, 0
    env = {**os.environ, "ESOLANG_DATASET_PATH": merged_path, "OPENROUTER_API_KEY": "local",
           "OPENROUTER_TEMPERATURE": str(temperature)}
    print(f"\n===== official harness: {regime} (temperature {temperature}) =====", flush=True)
    t0 = time.time()
    with subprocess.Popen(["esolang-run", "--model", MODEL_KEY, "--language", "brainfuck",
                           "--regime", regime, "--difficulty", "all"],
                          cwd=REPO_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
        code = proc.wait()
    print(f"[{regime}] exit {code}, {state['calls']} model calls, {(time.time() - t0) / 60:.0f} min",
          flush=True)


run("zero_shot", 0.0)              # greedy: the directly comparable number
run("self_scaffolding", TEMPERATURE)
print("\nScores are in each harness summary above. 3+/80 beats 2.5%.")
