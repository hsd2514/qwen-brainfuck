"""Kaggle, both T4s: v7.1 SFT - teach character codes, then check on held-out tasks.

Continues from hd2514p/bf-qwen-1.5b-sft-v7-plan and pushes to
hd2514p/bf-qwen-1.5b-sft-v71-ascii. ~4,000 verified examples, 1 epoch.

Output stays readable: a loss chart with one status line, key INFO lines, and a
held-out results table at the end. Everything else (warnings, progress bars,
upload bars) is written to /kaggle/working/v71_full.log; if the run fails, the
last 40 lines of that log are printed.

Setup: Accelerator GPU T4 x2, Internet on, secret HF_TOKEN.
"""
!pip install -q --no-deps git+https://github.com/hsd2514/bf-gym.git
!pip install -q -U gymnasium transformers trl peft accelerate
!pip uninstall -y -q torchao

import json
import os
import subprocess
import time

import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
from kaggle_secrets import UserSecretsClient

assert torch.cuda.device_count() == 2, (
    f"expected 2 GPUs, found {torch.cuda.device_count()} - set Accelerator to GPU T4 x2")

env = {
    **os.environ,
    "HF_TOKEN": UserSecretsClient().get_secret("HF_TOKEN"),
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONWARNINGS": "ignore",
    "TORCH_CPP_LOG_LEVEL": "ERROR",
}
env.pop("CUDA_VISIBLE_DEVICES", None)

REPO_DIR = "/kaggle/working/EsolangBench"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/Lossfunk/EsolangBench.git", REPO_DIR], check=True)

SCRIPT = "/kaggle/working/train_v71_sft.py"
LOG = "/kaggle/working/v71_full.log"
open(SCRIPT, "w", encoding="utf-8").write(r'''"""v7.1 SFT: teach character codes explicitly, continuing from the v7 adapter.

The v8 diagnostic showed v7 writes plans like "7 x 12 = 84 makes 'j'" - the
shape of a plan without knowing that 'j' is 106. Every constant plan here
states the code of each character and the arithmetic to reach it:
    'j' is 106. 106 - 0 = 106 = 10 x 10 + 6: loop 10 times adding 10 to cell 1, then add 6, print
Constants are random strings over all printable characters, so memorizing a
few words cannot work. Replay tasks (input handling, reversal, digits) also
state codes where they matter ("'a' is 97 and 'A' is 65").

Every program is verified by the strict interpreter before it is used. The held-out
eval at the end uses strings and parameter values that never appear in training.

Output protocol for the notebook: lines starting with INFO / PROGRESS / EVAL /
DONE are shown; everything else goes to the log file.
"""
import json
import os
import random
import string
import sys
import time

from bf_gym.config import BFConfig
from bf_gym.interpreter import run_bf

BASE_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SOURCE_ADAPTER = "hd2514p/bf-qwen-1.5b-sft-v7-plan"
HUB_OUT = "hd2514p/bf-qwen-1.5b-sft-v71-ascii"
OUT_DIR = "/kaggle/working"
REPO_DIR = f"{OUT_DIR}/EsolangBench"
MAX_LEN = 1536
OFFICIAL_SHARE = 0.25
N_CONST, N_DIGIT, N_REPLAY = 2600, 300, 1300
EVAL_PER_CATEGORY = 24

STRICT = BFConfig(wrap_pointer=False, max_steps=50_000)
PRINTABLE = "".join(chr(c) for c in range(32, 127))
NEUTRAL = ("Solve this in Brainfuck. First write a short plan of what each cell holds, then "
           "the program.\n\nTitle: {title}\n{description}")


def extract_code(text):
    return text.rsplit("\nCODE\n", 1)[-1].strip() if "\nCODE\n" in text else text.strip()


def name(ch):
    return {" ": "' ' (space)", "\n": "newline"}.get(ch, repr(ch))


# --------------------------------------------------------------------------
# Constants with explicit character codes
# --------------------------------------------------------------------------

def factor(n):
    """n = a * b + r with small a + b + |r| (a, b >= 2)."""
    best = None
    for a in range(2, 16):
        b = round(n / a)
        if b < 2:
            continue
        r = n - a * b
        cost = a + b + abs(r)
        if best is None or cost < best[0]:
            best = (cost, a, b, r)
    return best[1:]


def add_delta(d):
    """Code + plan fragment that changes cell 1 by d. Pointer on cell 1, cell 0 is 0."""
    sign, word = ("+", "adding") if d > 0 else ("-", "subtracting")
    if abs(d) <= 12:
        step = f"add {d}" if d > 0 else f"subtract {-d}"
        return sign * abs(d), f"{step} to cell 1"
    a, b, r = factor(abs(d))
    r_signed = r if d > 0 else -r
    code = "<" + "+" * a + "[>" + sign * b + "<-]>" + ("+" * r_signed if r_signed >= 0 else "-" * -r_signed)
    tail = f", then add {r_signed}" if r_signed > 0 else (f", then subtract {-r_signed}" if r_signed < 0 else "")
    return code, f"loop {a} times on cell 0 {word} {b} to cell 1 ({a} x {b} = {a * b}){tail}"


def constant_solution(text):
    code, plan, cur = [">"], ["cell 0 is the loop counter, cell 1 holds the character to print"], 0
    for ch in text:
        v = ord(ch)
        d = v - cur
        if d == 0:
            plan.append(f"{name(ch)} is {v}, cell 1 already holds it, print")
        else:
            frag, words = add_delta(d)
            code.append(frag)
            plan.append(f"{name(ch)} is {v}. {v} - {cur} = {d}: {words}, print")
        code.append(".")
        cur = v
    code = "".join(code)
    # a first loop starts with "<": drop the useless "><" at the very start
    return (code[2:] if code.startswith("><") else code), plan


def const_task(rng, text):
    return ("Print Text", f"Print exactly {text!r} and nothing else.", [(b"", text.encode())],
            *constant_solution(text))


def random_text(rng):
    kind = rng.random()
    if kind < 0.45:
        return "".join(rng.choice(PRINTABLE) for _ in range(rng.randint(1, 4)))
    if kind < 0.8:
        words = ["".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 6)))
                 for _ in range(rng.randint(1, 2))]
        t = " ".join(words)
        return t.capitalize() if rng.random() < 0.5 else t
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(1, 6)))


def digit_task(rng):
    text = "".join(rng.choice(string.digits) for _ in range(rng.randint(1, 3)))
    code, plan = constant_solution(text)
    plan.insert(1, "digits are characters: '0' is 48, so digit n is 48 + n, not n")
    return ("Print Digits", f"Print exactly {text!r} and nothing else.", [(b"", text.encode())],
            code, plan)


# --------------------------------------------------------------------------
# Replay tasks (training parameter values only; held-out values below)
# --------------------------------------------------------------------------

TRAIN_SHIFT, HELD_SHIFT = [1, 2, 3, 5, -1, -2, -3], [4, -4]
TRAIN_SEP, HELD_SEP = list("-_*"), list(",#")
TRAIN_BAR, HELD_BAR = list("Xo"), list("Z")


def line(rng, alphabet, lo, hi, empty_ok=True):
    if empty_ok and rng.random() < 0.15:
        return b""
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi))).encode()


def build_in_cell1_with_counter2(v):
    """Pointer starts on cell 0; value v built in cell 1 using cell 2 as counter; ends on cell 1."""
    a, b, r = factor(v)
    code = ">>" + "+" * a + "[<" + "+" * b + ">-]<" + ("+" * r if r >= 0 else "-" * -r)
    return code, f"{a} x {b}" + (f" + {r}" if r > 0 else (f" - {-r}" if r < 0 else ""))


def replay_task(rng, held=False):
    kind = rng.choice(["shift_char", "upper_char", "echo_line", "upper_line", "shift_line",
                       "sep_line", "reverse", "count", "sum_digits", "add_k", "digit_bar"])
    n = 4
    if kind == "shift_char":
        k = rng.choice(HELD_SHIFT if held else TRAIN_SHIFT)
        word = "after" if k > 0 else "before"
        ins = [bytes([rng.randint(ord("f"), ord("t"))]) for _ in range(n)]
        return (f"Shift Letter", f"Read one letter and print the letter {abs(k)} places {word} it "
                "in the alphabet. Input: one lowercase letter between f and t.",
                [(i, bytes([i[0] + k])) for i in ins], "," + ("+" * k if k > 0 else "-" * -k) + ".",
                ["read the letter into cell 0",
                 f"letters are in order ('a' is 97, 'b' is 98), so {'adding' if k > 0 else 'subtracting'} "
                 f"{abs(k)} moves it {abs(k)} places", "print it"])
    if kind == "upper_char":
        ins = [bytes([rng.randint(97, 122)]) for _ in range(n)]
        return ("Upper Letter", "Read one lowercase letter and print it in uppercase. Input: one "
                "lowercase letter.", [(i, bytes([i[0] - 32])) for i in ins], "," + "-" * 32 + ".",
                ["read the letter into cell 0", "'a' is 97 and 'A' is 65, so uppercase is 32 less",
                 "subtract 32, print it"])
    if kind == "echo_line":
        ins = [line(rng, string.ascii_letters, 1, 6) for _ in range(n)]
        return ("Echo", "Read the whole input and print it back unchanged. Input: a line of 0 to 6 "
                "letters, which may be empty.", [(i, i) for i in ins], ",[.,]",
                ["read a char into cell 0", "end of input reads as 0, which ends the loop",
                 "while cell 0 is not 0: print it, read the next char"])
    if kind == "upper_line":
        ins = [line(rng, string.ascii_lowercase, 1, 6) for _ in range(n)]
        return ("Upper Line", "Read the whole input (lowercase letters) and print it in uppercase. "
                "Input: a line of 0 to 6 letters, which may be empty.",
                [(i, bytes(c - 32 for c in i)) for i in ins], ",[" + "-" * 32 + ".,]",
                ["read a char into cell 0", "'a' is 97 and 'A' is 65, so uppercase is 32 less",
                 "while not end of input: subtract 32, print, read the next char"])
    if kind == "shift_line":
        k = rng.choice([1, 2, 3, -1, -2])
        ins = [line(rng, "fghijklmnop", 1, 6) for _ in range(n)]
        return ("Shift Line", f"Read the whole input and print every character with its byte value "
                f"changed by {k:+d}. Input: a line of 0 to 6 letters, which may be empty.",
                [(i, bytes(c + k for c in i)) for i in ins],
                ",[" + ("+" * k if k > 0 else "-" * -k) + ".,]",
                ["read a char into cell 0",
                 f"while not end of input: {'add' if k > 0 else 'subtract'} {abs(k)}, print, read the next char"])
    if kind == "sep_line":
        sep = rng.choice(HELD_SEP if held else TRAIN_SEP)
        build, math = build_in_cell1_with_counter2(ord(sep))
        ins = [line(rng, string.ascii_uppercase, 1, 5) for _ in range(n)]
        return ("Separated", f"Read the whole input and print each character followed by {sep!r}. "
                "Input: a line of 0 to 5 letters, which may be empty.",
                [(i, b"".join(bytes([c]) + sep.encode() for c in i)) for i in ins],
                build + "<,[.>.<,]",
                [f"{name(sep)} is {ord(sep)} = {math}: build it in cell 1 using cell 2 as the loop counter",
                 "go to cell 0 and read a char",
                 "while not end of input: print cell 0, print cell 1, read the next char into cell 0"])
    if kind == "reverse":
        ins = [line(rng, string.ascii_lowercase, 1, 7, empty_ok=False) for _ in range(n)]
        return ("Reverse", "Read the whole input and print it backwards. Input: a line of 1 to 7 "
                "letters.", [(i, i[::-1]) for i in ins], ">,[>,]<[.<]",
                ["cell 0 stays 0 as an end marker", "read chars into cells 1 2 3 and so on until input ends",
                 "step back to the last char", "print and move left until reaching the marker"])
    if kind == "count":
        ins = [line(rng, string.ascii_lowercase, 1, 7, empty_ok=False) for _ in range(n)]
        return ("Length", "Read the whole input and print how many characters it has, as one digit. "
                "Input: a line of 1 to 7 letters.", [(i, str(len(i)).encode()) for i in ins],
                ",[>+<,]>" + "+" * 48 + ".",
                ["read a char into cell 0",
                 "while not end of input: add 1 to the counter in cell 1, read the next char",
                 "'0' is 48, so add 48 to the count to make its digit, print it"])
    if kind == "sum_digits":
        ins = []
        for _ in range(n):
            a = rng.randint(0, 9)
            ins.append(f"{a} {rng.randint(0, 9 - a)}".encode())
        return ("Add Digits", "Read two digits separated by a space and print their sum as one digit. "
                "The sum is at most 9.", [(i, str(i[0] - 48 + i[2] - 48).encode()) for i in ins],
                ",>,>,[-<<+>>]<<" + "-" * 48 + ".",
                ["read the first digit into cell 0, the space into cell 1, the second digit into cell 2",
                 "move cell 2 into cell 0: cell 0 now holds both codes added",
                 "each digit code is 48 + its value, so the sum carries an extra 48: subtract 48",
                 "print it"])
    if kind == "add_k":
        k = rng.choice([4] if held else [1, 2, 3])
        ins = [str(rng.randint(0, 9 - k)).encode() for _ in range(n)]
        return ("Add Constant", f"Read one digit and print that number plus {k}, as one digit. The "
                "result is at most 9.", [(i, str(i[0] - 48 + k).encode()) for i in ins],
                "," + "+" * k + ".",
                ["read the digit into cell 0 ('3' is 51)",
                 f"digit codes are in order, so adding {k} gives the digit {k} larger", "print it"])
    ch = rng.choice(HELD_BAR if held else TRAIN_BAR)
    build, math = build_in_cell1_with_counter2(ord(ch))
    ins = [str(rng.randint(0, 9)).encode() for _ in range(n)]
    return ("Digit Bar", f"Read one digit n and print the character {ch!r} exactly n times.",
            [(i, ch.encode() * (i[0] - 48)) for i in ins],
            build + "<," + "-" * 48 + "[>.<-]",
            [f"{name(ch)} is {ord(ch)} = {math}: build it in cell 1 using cell 2 as the loop counter",
             "go to cell 0, read the digit, subtract 48 to turn '0'-'9' into 0-9",
             "loop that many times: print cell 1, subtract 1 from cell 0"])


# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------

def accepted_kwargs(config_cls, kwargs, tag):
    """Drop (and report) options the installed trl/transformers version no longer accepts,
    so a renamed setting logs a line instead of crashing the run."""
    import dataclasses
    known = {f.name for f in dataclasses.fields(config_cls)}
    dropped = sorted(k for k in kwargs if k not in known)
    if dropped:
        tag("INFO", f"skipping options not supported by this {config_cls.__name__}: {dropped}")
    return {k: v for k, v in kwargs.items() if k in known}


def verified(tests, code):
    return all((lambda r: not r.error and r.output == out)(run_bf(code, stdin=inp, config=STRICT))
               for inp, out in tests)


def to_prompt(rng, title, description, doc_text):
    if doc_text is not None and rng.random() < OFFICIAL_SHARE:
        from esolang_bench.benchmarking.prompt_templates import build_zero_shot_prompts
        problem = {"id": f"T{rng.randint(0, 99999):05d}", "title": title, "description": description}
        system, msgs = build_zero_shot_prompts("brainfuck", doc_text, problem)
        return [{"role": "system", "content": system}, {"role": "user", "content": msgs[0]}]
    return [{"role": "user", "content": NEUTRAL.format(title=title, description=description)}]


def build_rows(doc_text, seed=0):
    rng = random.Random(seed)
    rows, texts, rejected = [], set(), 0
    makers = ([lambda: const_task(rng, random_text(rng))] * N_CONST + [lambda: digit_task(rng)] * N_DIGIT
              + [lambda: replay_task(rng)] * N_REPLAY)
    rng.shuffle(makers)
    for make in makers:
        title, description, tests, code, plan = make()
        if not verified(tests, code):
            rejected += 1
            continue
        if title.startswith("Print"):
            texts.add(tests[0][1])
        completion = "PLAN\n" + "\n".join(plan) + "\nCODE\n" + code
        rows.append({"prompt": to_prompt(rng, title, description, doc_text),
                     "completion": [{"role": "assistant", "content": completion}]})
    return rows, texts, rejected


def eval_tasks(train_texts, seed=12345):
    """Held-out: unseen strings, unseen replay parameters (shift 4/-4, separators ',#', bar 'Z', add 4)."""
    rng = random.Random(seed)
    cats = {"print 1-4 chars": [], "print words": [], "input tasks (held-out params)": []}
    while len(cats["print 1-4 chars"]) < EVAL_PER_CATEGORY:
        t = "".join(rng.choice(PRINTABLE) for _ in range(rng.randint(1, 4)))
        if t.encode() not in train_texts:
            cats["print 1-4 chars"].append(const_task(rng, t))
    while len(cats["print words"]) < EVAL_PER_CATEGORY:
        t = " ".join("".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 6)))
                     for _ in range(2))
        if t.encode() not in train_texts:
            cats["print words"].append(const_task(rng, t))
    while len(cats["input tasks (held-out params)"]) < EVAL_PER_CATEGORY:
        task = replay_task(rng, held=True)
        if task[0] in ("Shift Letter", "Separated", "Digit Bar", "Add Constant"):
            cats["input tasks (held-out params)"].append(task)
    return cats


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    import torch
    import torch.distributed as dist
    import datasets
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from transformers.utils import logging as hf_logging
    from trl import SFTConfig, SFTTrainer

    hf_logging.set_verbosity_error()
    datasets.disable_progress_bars()
    sys.path.insert(0, REPO_DIR)
    doc_text = open(f"{REPO_DIR}/esolang_bench/docs/brainfuck.md", encoding="utf-8").read()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0
    torch.cuda.set_device(local_rank)
    # is_bf16_supported() says True on T4 in new torch (emulated), but the trainer rejects
    # bf16 below Ampere - decide by compute capability instead
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def tag(kind, payload):
        if is_main:
            print(f"{kind} {payload if isinstance(payload, str) else json.dumps(payload)}", flush=True)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    rows, train_texts, rejected = build_rows(doc_text)
    lengths = [len(tokenizer(tokenizer.apply_chat_template(r["prompt"] + r["completion"],
                                                           tokenize=False))["input_ids"]) for r in rows]
    kept = [r for r, n in zip(rows, lengths) if n <= MAX_LEN]
    tag("INFO", f"data: {len(kept)} verified examples ({rejected} failed verification, "
                f"{len(rows) - len(kept)} too long, longest {max(lengths)} tokens) "
                f"in {time.time() - t0:.0f}s")
    sample = next(r for r in kept if "Print Text" in r["prompt"][-1]["content"])
    tag("INFO", "sample: " + sample["completion"][0]["content"].replace("\n", " | "))

    base = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=dtype, device_map={"": local_rank})
    model = PeftModel.from_pretrained(base, SOURCE_ADAPTER, is_trainable=True)
    tag("INFO", f"loaded {SOURCE_ADAPTER} on {world} GPU(s), dtype {dtype}")

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs and "loss" in logs:
                tag("PROGRESS", {"step": state.global_step, "max_steps": state.max_steps,
                                 "loss": float(logs["loss"]),
                                 "lr": float(logs.get("learning_rate", 0)),
                                 "elapsed": time.time() - t0})

    per_device = 4
    accum = max(16 // (per_device * world), 1)
    config_kwargs = dict(
        output_dir=f"{OUT_DIR}/sft-v71",
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        num_train_epochs=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        max_length=MAX_LEN,
        logging_steps=5,
        save_strategy="no",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        report_to="none",
        disable_tqdm=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**accepted_kwargs(SFTConfig, config_kwargs, tag)),
        train_dataset=Dataset.from_list(kept),
        processing_class=tokenizer,
        callbacks=[Progress()],
    )
    trainer.train()
    model = trainer.model
    tag("INFO", f"training done in {(time.time() - t0) / 60:.0f} min")

    if is_main:
        model.save_pretrained(f"{OUT_DIR}/v71-adapter")
        api = HfApi()
        api.create_repo(HUB_OUT, exist_ok=True)
        api.upload_folder(folder_path=f"{OUT_DIR}/v71-adapter", repo_id=HUB_OUT,
                          commit_message="v7.1 ascii-explicit SFT")
        tag("INFO", f"pushed https://huggingface.co/{HUB_OUT}")

    # held-out eval, greedy; each rank scores its shard, sums are all_reduced
    model.eval()
    tokenizer.padding_side = "left"
    cats = eval_tasks(train_texts)
    names = list(cats)
    solved = torch.zeros(len(names), device=model.device)
    total = torch.zeros(len(names), device=model.device)
    for ci, cname in enumerate(names):
        mine = cats[cname][local_rank::world]
        for i in range(0, len(mine), 8):
            batch = mine[i:i + 8]
            texts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": NEUTRAL.format(title=t[0], description=t[1])}],
                add_generation_prompt=True, tokenize=False) for t in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=512, do_sample=False, use_cache=True,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            for t, o in zip(batch, out):
                raw = tokenizer.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                ok = verified(t[2], extract_code(raw))
                solved[ci] += ok
                total[ci] += 1
                if i == 0 and t is batch[0]:
                    tag("EVAL", {"sample": cname, "task": t[1], "ok": ok, "response": raw[:600]})
    if dist.is_initialized():
        dist.all_reduce(solved)
        dist.all_reduce(total)
    for ci, cname in enumerate(names):
        tag("EVAL", {"category": cname, "solved": int(solved[ci]), "total": int(total[ci])})
    tag("DONE", f"https://huggingface.co/{HUB_OUT}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
''')

steps, losses, info, samples, results = [], [], [], [], []
status = "starting: installing, building data, loading model..."
t0 = time.time()


def render():
    clear_output(wait=True)
    for line in info[-8:]:
        print(line)
    if steps:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(steps, losses, color="#0D9488", lw=2)
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("v7.1 SFT loss")
        ax.grid(alpha=0.25, lw=0.6)
        plt.tight_layout()
        plt.show()
    print(status, flush=True)


render()
with open(LOG, "w", encoding="utf-8") as log, subprocess.Popen(
    ["accelerate", "launch", "--multi_gpu", "--num_processes", "2", SCRIPT],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
) as proc:
    for line in proc.stdout:
        log.write(line)
        if line.startswith("PROGRESS "):
            p = json.loads(line[9:])
            steps.append(p["step"]); losses.append(p["loss"])
            eta = p["elapsed"] / max(p["step"], 1) * (p["max_steps"] - p["step"])
            status = (f"step {p['step']}/{p['max_steps']}  loss {p['loss']:.4f}  "
                      f"lr {p['lr']:.1e}  elapsed {p['elapsed'] / 60:.0f}m  eta {eta / 60:.0f}m")
            render()
        elif line.startswith("INFO "):
            info.append(line[5:].rstrip())
            render()
        elif line.startswith("EVAL "):
            e = json.loads(line[5:])
            (results if "category" in e else samples).append(e)
            status = "evaluating on held-out tasks..."
            render()
        elif line.startswith("DONE "):
            info.append("done: " + line[5:].rstrip())
    code = proc.wait()

render()
if samples:
    print("\n=== one held-out sample per category ===")
    for s in samples:
        print(f"\n[{s['sample']}] {'PASS' if s['ok'] else 'FAIL'}  task: {s['task']}")
        print("  " + s["response"].strip().replace("\n", "\n  "))
if results:
    print("\n=== held-out results (greedy) ===")
    for r in results:
        print(f"  {r['category']:32} {r['solved']:3d}/{r['total']}  ({r['solved'] / r['total']:.0%})")
if code != 0:
    lines = open(LOG, encoding="utf-8").read().splitlines()
    # the launcher's own traceback at the end is noise; show the first real one
    start = next((i for i, l in enumerate(lines) if "Traceback" in l), max(len(lines) - 40, 0))
    print("\n=== FAILED - first error in the log ===\n" + "\n".join(lines[start:start + 45]))
else:
    print(f"\nfinished in {(time.time() - t0) / 60:.0f} min  (full log: {LOG})")
