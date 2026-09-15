"""Brainfuck training v7 (Colab): plan first, then write the Brainfuck.

Why plans. Brainfuck's arithmetic is trivial - the hard part is bookkeeping:
while emitting `>>[<<+>>-]<<` the model has to silently track which cell the
pointer is on and what every cell holds. That is where earlier models broke
(half-formed programs with mismatched brackets). v7 trains the model to first
write a short plan naming the cells, then the program. The model still writes
every Brainfuck character itself; no compiler is involved at inference.

Every plan is generated from the SAME parameters as its program (e.g. the loop
counts used to build a constant), so a plan can never describe a different
program than the one that follows it.

Output format the model learns:
    PLAN
    <plan lines>
    CODE
    <brainfuck>
At eval, only the part after CODE is sent to the grader - the plan's periods,
commas and hyphens are Brainfuck commands and would corrupt the program. This is
the same as a reasoning model returning its answer without its thinking, and
should be stated whenever results are reported.

Prompt styles are mixed so the model learns the task rather than one wrapper:
official zero_shot and self_scaffolding prompts (built by the harness's own
functions), the older bf-gym prompt, and a neutral wording. A further wording is
deliberately kept OUT of training for a format-robustness eval.

Carried over from v6: 1.5B base, pointer-safe reversal, composition families,
strict (wrap_pointer=False) verification of every program, SFT only.
"""
!pip install -q --no-deps git+https://github.com/hsd2514/bf-gym.git
!pip install -q gymnasium trl peft accelerate
# Colab ships torchao 0.10, which new PEFT refuses to import. Nothing here uses
# torchao, so remove it. First run only: Runtime > Restart session, then rerun.
!pip uninstall -y -q torchao

import collections
import os
import random
import re
import string
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from huggingface_hub import login
from IPython.display import clear_output
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from bf_gym.config import BFConfig
from bf_gym.interpreter import run_bf
from bf_gym.tasks import ConstantOutputFamily, TaskMaker, make_task_family

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUT_DIR = "/content"
REPO_DIR = f"{OUT_DIR}/EsolangBench"
HUB_SFT = "hd2514p/bf-qwen-1.5b-sft-v7-plan"
MAX_LEN = 1536

assert torch.cuda.is_available(), "No GPU - Runtime > Change runtime type > GPU"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)} ({VRAM_GB:.0f} GB) | dtype: {DTYPE}", flush=True)

try:
    from google.colab import userdata
    login(token=userdata.get("HF_TOKEN"))
except Exception:
    login()

# sys.path import, not pip install -e: a package installed mid-session is not
# importable in the same running kernel.
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/Lossfunk/EsolangBench.git", REPO_DIR], check=True)
sys.path.insert(0, REPO_DIR)
from esolang_bench.benchmarking.prompt_templates import (
    build_self_scaffolding_prompt,
    build_zero_shot_prompts,
)

DOC_TEXT = open(f"{REPO_DIR}/esolang_bench/docs/brainfuck.md", encoding="utf-8").read()
STRICT = BFConfig(wrap_pointer=False)


def extract_code(text):
    """What the grader receives: everything after the CODE marker."""
    return text.rsplit("\nCODE\n", 1)[-1] if "\nCODE\n" in text else text


# --------------------------------------------------------------------------
# Verified conditional-branch idiom (count_vowels_line)
# --------------------------------------------------------------------------

class _Asm:
    def __init__(self):
        self.pos = 0
        self.code = []

    def goto(self, i):
        d = i - self.pos
        self.code.append(">" * d if d >= 0 else "<" * -d)
        self.pos = i

    def at(self, i, s):
        self.goto(i)
        self.code.append(s)

    def out(self):
        return "".join(self.code)


def _copy_nondestructive(a, src, dst, tmp):
    a.at(src, "["); a.at(dst, "+"); a.at(tmp, "+"); a.at(src, "-"); a.at(src, "]")
    a.at(tmp, "["); a.at(src, "+"); a.at(tmp, "-"); a.at(tmp, "]")


def _eq_test_const(a, x, k, temp0, temp1, y_const):
    """x = 1 if x == k else 0. y_const must be cleared first: the idiom restores y
    rather than draining it, so skipping the clear corrupts later comparisons."""
    a.at(y_const, "[-]"); a.at(y_const, "+" * k)
    a.at(temp0, "[-]"); a.at(temp1, "[-]")
    a.at(x, "["); a.at(temp1, "+"); a.at(x, "-"); a.at(x, "]")
    a.at(x, "+")
    a.at(y_const, "["); a.at(temp1, "-"); a.at(temp0, "+"); a.at(y_const, "-"); a.at(y_const, "]")
    a.at(temp0, "["); a.at(y_const, "+"); a.at(temp0, "-"); a.at(temp0, "]")
    a.at(temp1, "["); a.at(x, "-"); a.at(temp1, "[-]"); a.at(temp1, "]")


def _vowel_fragment():
    a = _Asm()
    a.at(1, "[-]")
    _copy_nondestructive(a, src=0, dst=1, tmp=2)
    a.at(1, "-" * 97)
    for off in (0, 4, 8, 14, 20):
        a.at(3, "[-]")
        _copy_nondestructive(a, src=1, dst=3, tmp=2)
        _eq_test_const(a, x=3, k=off, temp0=5, temp1=6, y_const=4)
        a.at(3, "["); a.at(7, "+"); a.at(3, "-]")
    a.goto(0)
    return a.out()


_VOWEL_FRAGMENT = _vowel_fragment()


def count_vowels_code(_task):
    a = _Asm()
    a.at(0, ",")
    a.at(0, "[")
    a.code.append(_VOWEL_FRAGMENT)
    a.at(7, "["); a.at(8, "+"); a.at(7, "-]")
    a.at(0, ",")
    a.at(0, "]")
    a.at(8, "+" * 48 + ".")
    return a.out()


COUNT_VOWELS_PLAN = [
    "read a char into cell 0",
    "while it is not end of input:",
    "  copy it to cell 1 and subtract 97 so that a is 0",
    "  compare cell 1 with 0 4 8 14 20 (a e i o u) using cells 2 to 6 as scratch",
    "  set cell 7 if any comparison matched",
    "  if cell 7 is set add 1 to the vowel counter in cell 8",
    "  read the next char into cell 0",
    "go to cell 8, add 48 to turn the count into a digit, print it",
]


# --------------------------------------------------------------------------
# Constants: code and plan built from the same construction
# --------------------------------------------------------------------------

def _value_parts(n):
    """Mirrors v6's _build_value exactly, but returns the parameters so the plan
    can describe the construction that was actually used."""
    best = parts = None
    for a in range(2, 24):
        for b in range(2, 24):
            r = n - a * b
            cost = a + b + abs(r) + 6
            if best is None or cost < best:
                best = cost
                parts = ("loop", a, b, r)
    if n + 1 < best:
        parts = ("direct", n)
    return parts


def _char(b):
    return repr(chr(b))


def constant_code_and_plan(target: bytes):
    code, plan, cur = [], [], None
    for b in target:
        if cur is None:
            parts = _value_parts(b)
            if parts[0] == "direct":
                code.append(">" + "+" * b)
                plan.append(f"go to cell 1 and add {b} to make {_char(b)}, print it")
            else:
                _, a, m, r = parts
                code.append("+" * a + "[>" + "+" * m + "<-]>" + ("+" * r if r >= 0 else "-" * -r))
                adjust = f" then add {r}" if r > 0 else (f" then subtract {-r}" if r < 0 else "")
                plan.append(f"set cell 0 to {a} and loop adding {m} to cell 1 each time "
                            f"({a * m}){adjust} to make {b} {_char(b)}, print it")
        else:
            d = b - cur
            code.append("+" * d if d >= 0 else "-" * -d)
            if d == 0:
                plan.append(f"cell 1 already holds {_char(b)}, print it again")
            else:
                step = f"add {d} to" if d > 0 else f"subtract {-d} from"
                plan.append(f"{step} cell 1 to make {_char(b)}, print it")
        code.append(".")
        cur = b
    return "".join(code), plan


# --------------------------------------------------------------------------
# Families: (family, code_fn, plan_fn)
# --------------------------------------------------------------------------

def _shift(n):
    return "+" * n if n >= 0 else "-" * -n


LOWER = string.ascii_lowercase
UPPER = string.ascii_uppercase
LETTERS = string.ascii_letters


def _str_sampler(lo, hi, alphabet):
    return lambda r: "".join(r.choice(alphabet) for _ in range(r.randint(lo, hi))).encode()


def _empty_ok_sampler(lo, hi, alphabet):
    def sample(r):
        if r.random() < 0.15:
            return b""
        return "".join(r.choice(alphabet) for _ in range(r.randint(lo, hi))).encode()
    return sample


REVERSE = ">,[>,]<[.<]"   # pointer-safe: cell 0 stays 0 and stops the walk back
REVERSE_PLAN = [
    "cell 0 stays 0 as an end marker",
    "read chars into cells 1 2 3 and so on until input ends",
    "step back to the last char",
    "print and move left until reaching the marker",
]


def static(lines):
    return lambda task: list(lines)


def build_families():
    out = []
    add = lambda fam, code_fn, plan_fn: out.append((fam, code_fn, plan_fn))

    const_code = lambda t: constant_code_and_plan(t.expected)[0]
    const_plan = lambda t: constant_code_and_plan(t.expected)[1]
    for c in (65, 97, 48, 32, 33, 63, 46, 10):
        add(ConstantOutputFamily(f"print_b{c}", 1, bytes([c])), const_code, const_plan)
    rng = random.Random(0)
    words = sorted({"".join(rng.choice(LETTERS) for _ in range(rng.randint(2, 4)))
                    for _ in range(20)})
    for w in words:
        add(ConstantOutputFamily(f"print_{w}", 2, w.encode()), const_code, const_plan)

    add(make_task_family(
        "echo_char", 3, "Read a single character and print it unchanged.",
        expected_fn=lambda b: b, sample_fn=lambda r: bytes([r.randint(33, 126)]),
        input_shape="one printable character"),
        lambda t: ",.", static(["read the char into cell 0", "print it"]))

    for k in (1, 2, 3, 5, -1, -2, -3):
        word = "later" if k > 0 else "earlier"
        step = f"add {k}" if k > 0 else f"subtract {-k}"
        add(make_task_family(
            f"shift_char_{k}", 4,
            f"Read a single letter and print the letter {abs(k)} position(s) {word} "
            "in the alphabet.",
            expected_fn=(lambda k: lambda b: bytes([b[0] + k]))(k),
            sample_fn=(lambda k: lambda r: bytes([r.randint(ord("a") - min(k, 0),
                                                            ord("z") - max(k, 0))]))(k),
            input_shape="one lowercase letter"),
            (lambda k: lambda t: "," + _shift(k) + ".")(k),
            static(["read the letter into cell 0", f"{step} to move it in the alphabet", "print it"]))

    add(make_task_family(
        "to_upper_char", 4, "Read a single lowercase letter and print its uppercase form.",
        expected_fn=lambda b: bytes([b[0] - 32]),
        sample_fn=lambda r: bytes([r.randint(ord("a"), ord("z"))]),
        input_shape="one lowercase letter"),
        lambda t: "," + "-" * 32 + ".",
        static(["read the letter into cell 0", "subtract 32 to turn lowercase into uppercase", "print it"]))
    add(make_task_family(
        "to_lower_char", 4, "Read a single uppercase letter and print its lowercase form.",
        expected_fn=lambda b: bytes([b[0] + 32]),
        sample_fn=lambda r: bytes([r.randint(ord("A"), ord("Z"))]),
        input_shape="one uppercase letter"),
        lambda t: "," + "+" * 32 + ".",
        static(["read the letter into cell 0", "add 32 to turn uppercase into lowercase", "print it"]))

    add(make_task_family(
        "echo_line", 5,
        "Read a line of text (which may be empty) and echo it back exactly as "
        "received, preserving every character including spaces.",
        expected_fn=lambda b: b, sample_fn=_empty_ok_sampler(0, 8, LETTERS + " "),
        input_shape="a line of 0-8 letters and spaces"),
        lambda t: ",[.,]",
        static(["read a char into cell 0",
                "while it is not end of input: print it, read the next char into cell 0"]))

    add(make_task_family(
        "count_chars_line", 6,
        "Read a line of text (which may be empty) and output the number of "
        "characters in it as a single digit.",
        expected_fn=lambda b: str(len(b)).encode(), sample_fn=_empty_ok_sampler(0, 9, UPPER),
        input_shape="a line of 0-9 uppercase letters"),
        lambda t: ",[>+<,]>" + "+" * 48 + ".",
        static(["read a char into cell 0",
                "while it is not end of input: add 1 to the counter in cell 1, read the next char",
                "go to cell 1, add 48 to turn the count into a digit, print it"]))

    add(make_task_family(
        "upper_line", 6,
        "Read a line of lowercase letters and output the same line in uppercase.",
        expected_fn=lambda b: bytes(c - 32 for c in b), sample_fn=_empty_ok_sampler(1, 6, LOWER),
        input_shape="a line of 1-6 lowercase letters"),
        lambda t: ",[" + "-" * 32 + ".,]",
        static(["read a char into cell 0",
                "while it is not end of input: subtract 32 to make it uppercase, print it, "
                "read the next char"]))

    add(make_task_family(
        "lower_line", 6,
        "Read a line of uppercase letters and output the same line in lowercase.",
        expected_fn=lambda b: bytes(c + 32 for c in b), sample_fn=_empty_ok_sampler(1, 6, UPPER),
        input_shape="a line of 1-6 uppercase letters"),
        lambda t: ",[" + "+" * 32 + ".,]",
        static(["read a char into cell 0",
                "while it is not end of input: add 32 to make it lowercase, print it, "
                "read the next char"]))

    add(make_task_family(
        "double_line", 6,
        "Read a line of text (which may be empty) and print each character twice.",
        expected_fn=lambda b: b"".join(bytes([c]) * 2 for c in b),
        sample_fn=_empty_ok_sampler(0, 5, UPPER),
        input_shape="a line of 0-5 uppercase letters"),
        lambda t: ",[..,]",
        static(["read a char into cell 0",
                "while it is not end of input: print it twice, read the next char"]))

    add(make_task_family(
        "shift_line", 6,
        "Read a line of lowercase letters and print each character shifted one "
        "position later in byte value.",
        expected_fn=lambda b: bytes(c + 1 for c in b), sample_fn=_empty_ok_sampler(1, 5, "abcxyw"),
        input_shape="a line of 1-5 lowercase letters"),
        lambda t: ",[+.,]",
        static(["read a char into cell 0",
                "while it is not end of input: add 1, print it, read the next char"]))

    add(make_task_family(
        "first_char_line", 5,
        "Read a line of text (which is never empty) and print only its first character.",
        expected_fn=lambda b: bytes([b[0]]), sample_fn=_str_sampler(1, 6, LETTERS),
        input_shape="a line of 1-6 letters"),
        lambda t: ",.",
        static(["read the first char into cell 0", "print it and stop"]))

    add(make_task_family(
        "count_vowels_line", 7,
        "Read a line of lowercase letters (0-3 characters) and output how many of "
        "them are vowels (a, e, i, o, u), as a single digit.",
        expected_fn=lambda b: str(sum(1 for c in b if c in b"aeiou")).encode(),
        sample_fn=_empty_ok_sampler(0, 3, LOWER),
        input_shape="a line of 0-3 lowercase letters"),
        count_vowels_code, static(COUNT_VOWELS_PLAN))

    add(make_task_family(
        "reverse_line", 6,
        "Read a line of text (which may be empty) and print it with the characters "
        "in reverse order.",
        expected_fn=lambda b: b[::-1], sample_fn=_empty_ok_sampler(0, 8, LETTERS + " "),
        input_shape="a line of 0-8 letters and spaces"),
        lambda t: REVERSE, static(REVERSE_PLAN))

    # compositions: transforms applied while storing, so no stored cell may become
    # 0 or the walk back stops early - alphabets are chosen so c-32 / c+1 never hit 0
    add(make_task_family(
        "upper_then_reverse", 7,
        "Read a line of lowercase letters and print it in uppercase with the "
        "characters in reverse order.",
        expected_fn=lambda b: bytes(c - 32 for c in b)[::-1],
        sample_fn=_empty_ok_sampler(1, 6, LOWER),
        input_shape="a line of 1-6 lowercase letters"),
        lambda t: ">,[" + "-" * 32 + ">,]<[.<]",
        static(["cell 0 stays 0 as an end marker",
                "read chars into cells 1 2 3 and so on, subtracting 32 from each as it is "
                "stored to make it uppercase",
                "step back to the last char",
                "print and move left until reaching the marker"]))

    add(make_task_family(
        "shift_then_reverse", 7,
        "Read a line of lowercase letters and print each character shifted one "
        "position later in byte value, in reverse order.",
        expected_fn=lambda b: bytes(c + 1 for c in b)[::-1],
        sample_fn=_empty_ok_sampler(1, 5, "abcxyw"),
        input_shape="a line of 1-5 lowercase letters"),
        lambda t: ">,[+>,]<[.<]",
        static(["cell 0 stays 0 as an end marker",
                "read chars into cells 1 2 3 and so on, adding 1 to each as it is stored",
                "step back to the last char",
                "print and move left until reaching the marker"]))

    add(make_task_family(
        "reverse_double", 7,
        "Read a line of text (which may be empty) and print it reversed, with "
        "every character printed twice.",
        expected_fn=lambda b: b"".join(bytes([c]) * 2 for c in b[::-1]),
        sample_fn=_empty_ok_sampler(0, 5, UPPER),
        input_shape="a line of 0-5 uppercase letters"),
        lambda t: ">,[>,]<[..<]",
        static(["cell 0 stays 0 as an end marker",
                "read chars into cells 1 2 3 and so on until input ends",
                "step back to the last char",
                "print each cell twice and move left until reaching the marker"]))

    add(make_task_family(
        "reverse_then_newline", 7,
        "Read a line of text (which may be empty) and print it reversed, followed "
        "by a newline.",
        expected_fn=lambda b: b[::-1] + b"\n", sample_fn=_empty_ok_sampler(0, 8, LETTERS + " "),
        input_shape="a line of 0-8 letters and spaces"),
        lambda t: REVERSE + "+" * 10 + ".",
        static(REVERSE_PLAN + ["the pointer is now on the marker, which is 0: add 10 and print "
                               "it as a newline"]))

    add(make_task_family(
        "sum_two_tokens", 7,
        "Read two single-digit numbers separated by a space and print their sum "
        "as a single digit (the sum never exceeds 9).",
        expected_fn=lambda b: str((b[0] - 48) + (b[2] - 48)).encode(),
        sample_fn=lambda r: (lambda a, c: f"{a} {c}".encode())(
            *[(x, y) for x in range(10) for y in range(10) if x + y <= 9][r.randrange(55)]),
        input_shape="two digits separated by a space"),
        lambda t: ",>,>,[-<<+>>]<<" + "-" * 48 + ".",
        static(["read the first digit into cell 0, the space into cell 1, the second digit "
                "into cell 2",
                "move the value of cell 2 into cell 0, so cell 0 holds both digit codes added",
                "the two codes each carry 48, so subtract 48 to leave one digit character",
                "print it"]))

    return out


ALL_FAMILIES = build_families()
CODE_FNS = {f.name: c for f, c, _ in ALL_FAMILIES}
PLAN_FNS = {f.name: p for f, _, p in ALL_FAMILIES}


def describe(task):
    """Official problems describe the task and input in prose and never show
    concrete input bytes, so neither do these descriptions."""
    d = task.family.description(task.stdin)
    if task.family.reads_input():
        d += f" Input: {task.family.input_shape_description()}."
    else:
        d += f" The exact output is {task.expected.decode('latin-1')!r}."
    return d


# --------------------------------------------------------------------------
# Prompt styles
# --------------------------------------------------------------------------

NEUTRAL_TEMPLATE = ("Solve this in Brainfuck. First write a short plan of what each cell "
                    "holds, then the program.\n\nTitle: {title}\n{description}")

# Never used in training - reserved for the format-robustness eval, so a gain
# that only shows up in trained phrasings can be told apart from real skill.
HELDOUT_TEMPLATE = ("Here is a programming task. Your answer must be a Brainfuck program."
                    "\n\n{description}")


def make_prompt(task, problem, rng):
    x = rng.random()
    if x < 0.3:
        system, msgs = build_zero_shot_prompts("brainfuck", DOC_TEXT, problem)
        return "official_zero_shot", [{"role": "system", "content": system},
                                      {"role": "user", "content": msgs[0]}]
    if x < 0.6:
        system, msgs = build_self_scaffolding_prompt("brainfuck", DOC_TEXT, problem, None, None)
        return "official_self_scaffolding", [{"role": "system", "content": system},
                                             {"role": "user", "content": msgs[0]}]
    if x < 0.8:
        return "bf_gym", [{"role": "user", "content": task.prompt}]
    return "neutral", [{"role": "user", "content": NEUTRAL_TEMPLATE.format(**problem)}]


def completion_text(plan_lines, code):
    return "PLAN\n" + "\n".join(plan_lines) + "\nCODE\n" + code


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def build_sft_dataset(seed=0):
    rng = random.Random(seed)
    rows, styles, attempted, rejected = [], collections.Counter(), 0, 0
    small = [f for f, _, _ in ALL_FAMILIES if f.difficulty <= 4]
    unbounded = [f for f, _, _ in ALL_FAMILIES if f.difficulty >= 5]
    for families, n_each in ((small, 15), (unbounded, 110)):
        maker = TaskMaker(families, seed=seed)
        for _ in range(n_each * len(families)):
            task = maker.next_task()
            attempted += 1
            code = CODE_FNS[task.family.name](task)
            plan = PLAN_FNS[task.family.name](task)
            text = completion_text(plan, code)
            result = run_bf(extract_code(text), stdin=task.stdin, config=STRICT)
            if result.error or result.output != task.expected or extract_code(text) != code:
                rejected += 1
                continue
            problem = {
                "id": f"T{len(rows) + 1:05d}",
                "title": task.family.name.replace("_", " ").title(),
                "description": describe(task),
            }
            style, prompt = make_prompt(task, problem, rng)
            styles[style] += 1
            rows.append({"prompt": prompt,
                         "completion": [{"role": "assistant", "content": text}]})

    lengths = [
        len(tokenizer(tokenizer.apply_chat_template(r["prompt"] + r["completion"],
                                                    tokenize=False))["input_ids"])
        for r in rows
    ]
    kept = [r for r, n in zip(rows, lengths) if n <= MAX_LEN]
    print(f"SFT: {len(kept)} verified plan+code pairs | attempted {attempted}, rejected "
          f"{rejected}, dropped for length {len(rows) - len(kept)} (max {max(lengths)} "
          f"tokens, limit {MAX_LEN})", flush=True)
    print("prompt styles:", dict(styles), flush=True)
    return Dataset.from_list(kept)


sft_dataset = build_sft_dataset()
print("\nsample completion:\n" + sft_dataset[0]["completion"][0]["content"], flush=True)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

class ProgressCallback(TrainerCallback):
    """clear_output() wipes tqdm, so step/ETA are printed with the chart."""

    def __init__(self, plot_every=20):
        self.plot_every = plot_every
        self.t0 = None
        self.steps, self.losses = [], []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        if self.t0 is None:
            self.t0 = time.time()
        self.steps.append(state.global_step)
        self.losses.append(logs["loss"])
        if state.global_step % self.plot_every == 0:
            self.draw(state.max_steps)

    def draw(self, total):
        elapsed = time.time() - self.t0
        eta = elapsed / max(len(self.steps), 1) * max(total - self.steps[-1], 0)
        clear_output(wait=True)
        plt.figure(figsize=(8, 3))
        plt.plot(self.steps, self.losses, color="#0D9488", lw=2)
        plt.xlabel("step"); plt.ylabel("loss"); plt.title("SFT loss")
        plt.grid(alpha=0.25, lw=0.6)
        plt.tight_layout()
        plt.show()
        print(f"step {self.steps[-1]}/{total}  loss {self.losses[-1]:.4f}  "
              f"elapsed {elapsed / 60:.0f}m  eta {eta / 60:.0f}m", flush=True)


if VRAM_GB >= 40:
    batch, accum = 8, 2
elif VRAM_GB >= 22:
    batch, accum = 4, 4
else:
    batch, accum = 2, 8

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map={"": 0})

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=f"{OUT_DIR}/sft-v7",
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        gradient_checkpointing=True,
        num_train_epochs=2,
        learning_rate=2e-4,
        max_length=MAX_LEN,
        logging_steps=5,
        save_strategy="no",
        bf16=DTYPE == torch.bfloat16,
        report_to="none",
        push_to_hub=True,
        hub_model_id=HUB_SFT,
    ),
    train_dataset=sft_dataset,
    processing_class=tokenizer,
    peft_config=LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ),
)
trainer.add_callback(ProgressCallback())
trainer.train()
trainer.save_model(f"{OUT_DIR}/bf-qwen-1.5b-sft-v7-plan")
trainer.push_to_hub()

print("done:", f"https://huggingface.co/{HUB_SFT}")
print("For eval: base model 1.5B, adapter", HUB_SFT, "- the eval server must return only "
      "extract_code(response) to the grader and allow ~768 new tokens (plan + code).")
