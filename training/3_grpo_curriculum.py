"""Kaggle, both T4s: v8 GRPO curriculum, reward = interpreter only.

Starts from hd2514p/bf-qwen-1.5b-sft-v71-ascii. 8 stages, one Brainfuck concept
each; the model never sees a solution, only whether its code passed hidden
tests. Moves on when it solves >= 60% of HELD-OUT task variants (parameter
values never trained on), at most 3 rounds per stage.

Pushes to hd2514p/bf-qwen-1.5b-grpo-v8b after every round. If the session dies,
rerun this cell: it resumes from the last pushed stage/round.

Setup: Accelerator GPU T4 x2, Internet on, secret HF_TOKEN.
If PEFT complains about torchao, restart the session once and rerun.
"""
!pip install -q --no-deps git+https://github.com/hsd2514/bf-gym.git
# -U transformers: newest peft imports transformers features Kaggle's copy lacks
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
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"   # hide the per-round upload bars
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from huggingface_hub import file_exists
START_ADAPTER = "hd2514p/bf-qwen-1.5b-sft-v71-ascii"
if not file_exists(START_ADAPTER, "adapter_config.json", token=os.environ["HF_TOKEN"]):
    raise SystemExit(f"{START_ADAPTER} is not on the Hub yet - run the v7.1 SFT cell first, "
                     "then rerun this cell.")

REPO_DIR = "/kaggle/working/EsolangBench"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/Lossfunk/EsolangBench.git", REPO_DIR], check=True)

SCRIPT = "/kaggle/working/train_v8_grpo.py"
open(SCRIPT, "w", encoding="utf-8").write(r'''"""v8: GRPO curriculum from the v7 SFT model, graded only by the interpreter.

Why RL now. v7 learned ~35 program templates and, for anything new, emits the
closest one (count-chars for word count, etc). Supervised data can only show more
templates. Here the model is never shown a solution: it writes PLAN + CODE, the
interpreter runs the code on freshly generated hidden tests, and the reward is
the fraction passed. Nothing else is rewarded - no similarity, no length bonus.

Curriculum: 8 stages, one concept each. Every stage has many task variants
(different constants, separators, letters...) so no single program wins, and
some parameter values are held out of training entirely. After each round the
model is scored greedily on held-out variants; it moves to the next stage at
>= GATE solved, else trains another round (at most MAX_ROUNDS, then moves on
anyway so one hard stage cannot stall the run). 20% of each round replays
earlier stages so skills are not forgotten.

The adapter and curriculum position are pushed to the Hub after every round, so
an interrupted Kaggle session resumes where it stopped.

Launched with `accelerate launch` (one process per GPU). All ranks build the
same datasets from the same seeds and reach the same gate decision because the
held-out score is summed across ranks with all_reduce.
"""
import gc
import json
import os
import random
import string

from bf_gym.config import BFConfig
from bf_gym.interpreter import run_bf

BASE_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SFT_ADAPTER = "hd2514p/bf-qwen-1.5b-sft-v71-ascii"
HUB_RL = "hd2514p/bf-qwen-1.5b-grpo-v8b"   # fresh repo: the old one holds the failed run's state
OUT_DIR = "/kaggle/working"
REPO_DIR = f"{OUT_DIR}/EsolangBench"

STEPS_PER_ROUND = 20
MAX_ROUNDS = 3
GATE = 0.6
HELDOUT_PER_STAGE = 16
OFFICIAL_SHARE = 0.1   # the official prompt carries ~800 tokens of docs: keep it rare for speed
REPLAY = 0.2
NUM_GENERATIONS = 4
# v7.1 plans state every character's code: a 2-word phrase needs up to ~670 tokens,
# and a truncated answer earns no reward at all
MAX_COMPLETION = 768

STRICT = BFConfig(wrap_pointer=False, max_steps=20_000)
LOWER, UPPER, DIGITS = string.ascii_lowercase, string.ascii_uppercase, string.digits


def extract_code(text):
    return text.rsplit("\nCODE\n", 1)[-1].strip() if "\nCODE\n" in text else text.strip()


# --------------------------------------------------------------------------
# Task variants. Each maker(rng, heldout) returns (title, description,
# input_sampler or None, output_fn). `heldout` picks parameter values that
# never appear in training.
# --------------------------------------------------------------------------

def pick(rng, heldout, train_vals, held_vals):
    return rng.choice(held_vals if heldout else train_vals)


def line(rng, alphabet, lo, hi, empty_ok=True):
    if empty_ok and rng.random() < 0.15:
        return b""
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi))).encode()


def s1_print_short(rng, heldout):
    pool = "nopqrstuvwxyzNOPQRSTUVWXYZ!?" if heldout else "abcdefghijklmABCDEFGHIJKLM0123456789.:"
    text = "".join(rng.choice(pool) for _ in range(rng.randint(1, 3)))
    return "Print Text", f"Print exactly {text!r} and nothing else.", None, lambda b: text.encode()


def s2_print_words(rng, heldout):
    words = (["river", "stone", "cloud", "maple", "orbit"] if heldout
             else ["hello", "world", "brain", "code", "tape", "cell", "loop", "byte"])
    # mostly one word: two-word answers run ~670 tokens and every batch waits for its longest
    text = " ".join(rng.choice(words) for _ in range(1 if rng.random() < 0.7 else 2))
    if rng.random() < 0.5:
        text = text.capitalize()
    if rng.random() < 0.3:
        text += "!"
    return "Print Phrase", f"Print exactly {text!r} with no newline.", None, lambda b: text.encode()


def s3_char(rng, heldout):
    kind = rng.choice(["shift", "twice", "newline", "next_pair", "upper"])
    one = lambda r: bytes([r.randint(ord("f"), ord("t"))])
    shape = " Input: one lowercase letter between f and t."
    if kind == "shift":
        k = pick(rng, heldout, [1, 2, 3, 5, -1, -2, -3, -5], [4, -4])
        word = "after" if k > 0 else "before"
        return ("Shift Letter", f"Read one letter and print the letter {abs(k)} places {word} "
                f"it in the alphabet." + shape, one, lambda b: bytes([b[0] + k]))
    if kind == "twice":
        n = pick(rng, heldout, [2, 3], [4])
        return ("Repeat Letter", f"Read one letter and print it {n} times." + shape, one,
                lambda b: b * n)
    if kind == "newline":
        return ("Letter Line", "Read one letter and print it followed by a newline." + shape, one,
                lambda b: b + b"\n")
    if kind == "next_pair":
        return ("Letter Pair", "Read one letter and print it followed by the next letter of "
                "the alphabet." + shape, one, lambda b: b + bytes([b[0] + 1]))
    return ("Upper Letter", "Read one lowercase letter and print it in uppercase." + shape, one,
            lambda b: bytes([b[0] - 32]))


def s4_line_loop(rng, heldout):
    kind = rng.choice(["echo", "shift", "upper", "lower", "double", "sep"])
    shape = " Input: a line of 0 to 6 letters, which may be empty."
    if kind == "echo":
        return ("Echo", "Read the whole input and print it back unchanged." + shape,
                lambda r: line(r, LOWER + UPPER, 1, 6), lambda b: b)
    if kind == "shift":
        k = pick(rng, heldout, [1, 2, 3, -1, -2], [4, -3])
        return ("Shift Line", f"Read the whole input and print every character with its byte "
                f"value changed by {k:+d}." + shape, lambda r: line(r, "fghijklmnop", 1, 6),
                lambda b: bytes(c + k for c in b))
    if kind == "upper":
        return ("Upper Line", "Read the whole input (lowercase letters) and print it in "
                "uppercase." + shape, lambda r: line(r, LOWER, 1, 6),
                lambda b: bytes(c - 32 for c in b))
    if kind == "lower":
        return ("Lower Line", "Read the whole input (uppercase letters) and print it in "
                "lowercase." + shape, lambda r: line(r, UPPER, 1, 6),
                lambda b: bytes(c + 32 for c in b))
    if kind == "double":
        n = pick(rng, heldout, [2], [3])
        return ("Stretch Line", f"Read the whole input and print each character {n} times." + shape,
                lambda r: line(r, UPPER, 1, 5), lambda b: b"".join(bytes([c]) * n for c in b))
    sep = pick(rng, heldout, list("-_*"), list(",#"))
    return ("Separated", f"Read the whole input and print each character followed by "
            f"{sep!r}." + shape, lambda r: line(r, UPPER, 1, 5),
            lambda b: b"".join(bytes([c]) + sep.encode() for c in b))


def s5_memory(rng, heldout):
    kind = rng.choice(["count", "reverse", "reverse_shift", "last", "twice", "stars"])
    shape = " Input: a line of 1 to 7 letters."
    nonempty = lambda r: line(r, LOWER, 1, 7, empty_ok=False)
    if kind == "count":
        return ("Length", "Read the whole input and print how many characters it has, as one "
                "digit." + shape, nonempty, lambda b: str(len(b)).encode())
    if kind == "reverse":
        return ("Reverse", "Read the whole input and print it backwards." + shape, nonempty,
                lambda b: b[::-1])
    if kind == "reverse_shift":
        k = pick(rng, heldout, [1, 2, -1], [3])
        return ("Reverse Shift", f"Read the whole input and print it backwards with every byte "
                f"value changed by {k:+d}." + shape, nonempty,
                lambda b: bytes(c + k for c in b[::-1]))
    if kind == "last":
        return ("Last Char", "Read the whole input and print only its last character." + shape,
                nonempty, lambda b: b[-1:])
    if kind == "twice":
        return ("Line Twice", "Read the whole input and print it two times in a row." + shape,
                nonempty, lambda b: b + b)
    star = pick(rng, heldout, ["*", "#"], ["@"])
    return ("Length Bar", f"Read the whole input and print one {star!r} per character." + shape,
            nonempty, lambda b: star.encode() * len(b))


def digits_pair(r, ok):
    pairs = [(a, c) for a in range(10) for c in range(10) if ok(a, c)]
    a, c = r.choice(pairs)
    return f"{a} {c}".encode()


def s6_arith(rng, heldout):
    kind = rng.choice(["sum", "diff", "add_k", "double", "bar"])
    if kind == "sum":
        return ("Add Digits", "Read two digits separated by a space and print their sum as one "
                "digit. The sum is at most 9.", lambda r: digits_pair(r, lambda a, c: a + c <= 9),
                lambda b: str(b[0] - 48 + b[2] - 48).encode())
    if kind == "diff":
        return ("Subtract Digits", "Read two digits separated by a space and print the first "
                "minus the second. The first is never smaller.",
                lambda r: digits_pair(r, lambda a, c: a >= c),
                lambda b: str(b[0] - b[2]).encode())
    if kind == "add_k":
        k = pick(rng, heldout, [1, 2, 3], [4])
        return ("Add Constant", f"Read one digit and print that number plus {k}, as one digit. "
                "The result is at most 9.", lambda r: str(r.randint(0, 9 - k)).encode(),
                lambda b: str(b[0] - 48 + k).encode())
    if kind == "double":
        return ("Double Digit", "Read one digit and print twice its value, as one digit. The "
                "result is at most 9.", lambda r: str(r.randint(0, 4)).encode(),
                lambda b: str(2 * (b[0] - 48)).encode())
    ch = pick(rng, heldout, ["X", "o"], ["Z"])
    return ("Digit Bar", f"Read one digit n and print the character {ch!r} exactly n times.",
            lambda r: str(r.randint(0, 9)).encode(), lambda b: ch.encode() * (b[0] - 48))


def s7_conditional(rng, heldout):
    kind = rng.choice(["count_letter", "remove_letter", "until_space", "replace_letter"])
    letter = pick(rng, heldout, list("aeiost"), list("nr"))
    words = lambda r: line(r, "aeinorst", 1, 7, empty_ok=True)
    shape = " Input: a line of 0 to 7 lowercase letters."
    if kind == "count_letter":
        return ("Count Letter", f"Read the whole input and print how many times the letter "
                f"{letter!r} appears, as one digit." + shape, words,
                lambda b: str(b.count(letter.encode())).encode())
    if kind == "remove_letter":
        return ("Remove Letter", f"Read the whole input and print it with every {letter!r} "
                "removed." + shape, words, lambda b: b.replace(letter.encode(), b""))
    if kind == "until_space":
        return ("First Word", "Read the whole input and print only the part before the first "
                "space (all of it if there is no space). Input: two short lowercase words "
                "separated by a space, or one word.",
                lambda r: line(r, "abcde", 1, 4, False) + (b" " + line(r, "abcde", 1, 4, False)
                                                           if r.random() < 0.7 else b""),
                lambda b: b.split(b" ")[0])
    other = pick(rng, heldout, list("xyz"), list("q"))
    return ("Replace Letter", f"Read the whole input and print it with every {letter!r} "
            f"replaced by {other!r}." + shape, words,
            lambda b: b.replace(letter.encode(), other.encode()))


def s8_combo(rng, heldout):
    kind = rng.choice(["reverse_upper", "count_then_echo", "upper_sep", "reverse_remove"])
    letter = pick(rng, heldout, list("aest"), list("o"))
    shape = " Input: a line of 1 to 6 lowercase letters."
    nonempty = lambda r: line(r, "aeostlm", 1, 6, empty_ok=False)
    if kind == "reverse_upper":
        return ("Reverse Upper", "Read the whole input and print it backwards in uppercase." + shape,
                nonempty, lambda b: bytes(c - 32 for c in b[::-1]))
    if kind == "count_then_echo":
        return ("Length Then Text", "Read the whole input, then print its length as one digit "
                "followed by the input itself." + shape, nonempty,
                lambda b: str(len(b)).encode() + b)
    if kind == "upper_sep":
        sep = pick(rng, heldout, ["-", "."], ["+"])
        return ("Upper Separated", f"Read the whole input and print each character in "
                f"uppercase followed by {sep!r}." + shape, nonempty,
                lambda b: b"".join(bytes([c - 32]) + sep.encode() for c in b))
    return ("Reverse Without", f"Read the whole input and print it backwards with every "
            f"{letter!r} removed." + shape, nonempty,
            lambda b: b[::-1].replace(letter.encode(), b""))


STAGES = [s1_print_short, s2_print_words, s3_char, s4_line_loop, s5_memory, s6_arith,
          s7_conditional, s8_combo]
STAGE_NAMES = ["print short", "print words", "one char", "line loops", "memory cells",
               "digit math", "conditionals", "combinations"]

NEUTRAL = ("Solve this in Brainfuck. First write a short plan of what each cell holds, then "
           "the program.\n\nTitle: {title}\n{description}")


def make_task(stage, rng, heldout, doc_text=None, n_tests=4):
    title, description, sampler, fn = STAGES[stage](rng, heldout)
    inputs = [b""] if sampler is None else [sampler(rng) for _ in range(n_tests)]
    tests = [[i.decode("latin-1"), fn(i).decode("latin-1")] for i in inputs]
    if doc_text is not None and rng.random() < OFFICIAL_SHARE:
        from esolang_bench.benchmarking.prompt_templates import build_zero_shot_prompts
        problem = {"id": f"R{rng.randint(0, 99999):05d}", "title": title, "description": description}
        system, msgs = build_zero_shot_prompts("brainfuck", doc_text, problem)
        prompt = [{"role": "system", "content": system}, {"role": "user", "content": msgs[0]}]
    else:
        prompt = [{"role": "user", "content": NEUTRAL.format(title=title, description=description)}]
    return {"prompt": prompt, "tests": json.dumps(tests), "stage": stage}


def pass_fraction(code, tests):
    tests = json.loads(tests)
    ok = 0
    for inp, out in tests:
        res = run_bf(code, stdin=inp.encode("latin-1"), config=STRICT)
        ok += (not res.error) and res.output == out.encode("latin-1")
    return ok / len(tests)


def _text(c):
    return c[0]["content"] if isinstance(c, list) else c


def reward_pass(prompts, completions, tests, **kw):
    return [pass_fraction(extract_code(_text(c)), t) for c, t in zip(completions, tests)]


def reward_format(prompts, completions, **kw):
    out = []
    for c in completions:
        text = _text(c)
        code = extract_code(text)
        r = 0.0
        if not (text.lstrip().startswith("PLAN") and "\nCODE\n" in text):
            r -= 0.5
        if any(ch not in "><+-.,[] \n" for ch in code) or code.count("[") != code.count("]"):
            r -= 0.25
        out.append(r)
    return out


def reward_solved(prompts, completions, tests, **kw):
    """Weight 0 - logged only, it is the number the curriculum gate watches."""
    return [float(p == 1.0) for p in reward_pass(prompts, completions, tests)]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def main():
    import torch
    import torch.distributed as dist
    from datasets import Dataset
    from huggingface_hub import HfApi, hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    import sys

    sys.path.insert(0, REPO_DIR)
    doc_text = open(f"{REPO_DIR}/esolang_bench/docs/brainfuck.md", encoding="utf-8").read()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0
    torch.cuda.set_device(local_rank)
    # No init_process_group here: loading a saved adapter while the process group
    # is up sends PEFT down its tensor-parallel path. The trainer starts the group.
    # is_bf16_supported() says True on T4 in new torch (emulated), but the trainer rejects
    # bf16 below Ampere - decide by compute capability instead
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def say(*a):
        if is_main:
            print(*a, flush=True)

    # resume from the Hub if a previous session got partway
    stage, rnd, history = 0, 0, []
    source = SFT_ADAPTER
    try:
        state = json.load(open(hf_hub_download(HUB_RL, "curriculum_state.json")))
        stage, rnd, history = state["stage"], state["round"], state["history"]
        source = HUB_RL
        say(f"RESUME stage {stage} round {rnd} from {HUB_RL}")
    except Exception:
        say(f"START from {SFT_ADAPTER}")
    for h in history:
        say("STAGE_EVAL " + json.dumps(h))

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=dtype, device_map={"": local_rank})
    model = PeftModel.from_pretrained(base, source, is_trainable=True)

    class Progress(TrainerCallback):
        def __init__(self, stage, rnd):
            self.stage, self.rnd = stage, rnd

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not (state.is_world_process_zero and logs and "reward" in logs):
                return
            pick_ = lambda k: next((v for kk, v in logs.items() if k in kk), None)
            print("PROGRESS " + json.dumps({
                "stage": self.stage, "round": self.rnd, "step": state.global_step,
                "max_steps": state.max_steps, "reward": logs.get("reward"),
                "solved": pick_("reward_solved/mean"), "pass": pick_("reward_pass/mean"),
                "zero_std": logs.get("frac_reward_zero_std"),
                "length": pick_("completions/mean_length"),
            }), flush=True)

    def heldout_score(stage_idx):
        """Greedy, on held-out variants; each rank scores its shard, sums are all_reduced."""
        rng = random.Random(10_000 + stage_idx)
        tasks = [make_task(stage_idx, rng, True, doc_text, n_tests=6)
                 for _ in range(HELDOUT_PER_STAGE)]
        mine = tasks[local_rank::world]
        tokenizer.padding_side = "left"
        model.eval()
        solved = 0
        with torch.no_grad():
            for i in range(0, len(mine), 8):
                batch = mine[i:i + 8]
                texts = [tokenizer.apply_chat_template(t["prompt"], add_generation_prompt=True,
                                                       tokenize=False) for t in batch]
                enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
                out = model.generate(**enc, max_new_tokens=MAX_COMPLETION, do_sample=False,
                                     use_cache=True,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                for t, o in zip(batch, out):
                    raw = tokenizer.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    solved += pass_fraction(extract_code(raw), t["tests"]) == 1.0
        model.train()
        tokenizer.padding_side = "right"
        counts = torch.tensor([solved, len(mine)], device=model.device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(counts)
        return counts[0].item() / counts[1].item()

    def push_state():
        if not is_main:
            return
        model.save_pretrained(f"{OUT_DIR}/v8-adapter")
        json.dump({"stage": stage, "round": rnd, "history": history},
                  open(f"{OUT_DIR}/v8-adapter/curriculum_state.json", "w"))
        api = HfApi()
        api.create_repo(HUB_RL, exist_ok=True)
        api.upload_folder(folder_path=f"{OUT_DIR}/v8-adapter", repo_id=HUB_RL,
                          commit_message=f"stage {stage} round {rnd}")

    per_device = 4
    accum = 2
    prompts_per_step = per_device * accum * world // NUM_GENERATIONS

    while stage < len(STAGES):
        # skip a stage the model already passes: a ~3 min check instead of a ~40 min round
        if rnd == 0:
            score = heldout_score(stage)
            history.append({"stage": stage, "round": 0, "heldout_solved": score})
            say("STAGE_EVAL " + json.dumps(history[-1]))
            if score >= GATE:
                say(f"stage {stage + 1} already passed before training ({score:.0%}), skipping")
                stage += 1
                push_state()
                continue
        rng = random.Random(1_000 * stage + rnd)
        n_rows = STEPS_PER_ROUND * prompts_per_step * 2
        rows = []
        for _ in range(n_rows):
            s = rng.randrange(stage) if stage > 0 and rng.random() < REPLAY else stage
            rows.append(make_task(s, rng, False, doc_text))
        say(f"\n=== stage {stage + 1}/{len(STAGES)} ({STAGE_NAMES[stage]}) round {rnd + 1} ===")

        config_kwargs = dict(
            output_dir=f"{OUT_DIR}/grpo-v8",
            max_steps=STEPS_PER_ROUND,
            per_device_train_batch_size=per_device,
            gradient_accumulation_steps=accum,
            num_generations=NUM_GENERATIONS,
            max_completion_length=MAX_COMPLETION,
            temperature=1.0,
            learning_rate=1e-5,
            reward_weights=[1.0, 1.0, 0.0],
            mask_truncated_completions=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            logging_steps=1,
            disable_tqdm=True,   # its "\r" redraws were hiding the PROGRESS lines from the chart
            save_strategy="no",
            report_to="none",
            seed=stage * 100 + rnd,
        )
        import dataclasses
        known = {f.name for f in dataclasses.fields(GRPOConfig)}
        dropped = sorted(k for k in config_kwargs if k not in known)
        if dropped:
            say(f"skipping options not supported by this GRPOConfig: {dropped}")
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_pass, reward_format, reward_solved],
            args=GRPOConfig(**{k: v for k, v in config_kwargs.items() if k in known}),
            train_dataset=Dataset.from_list(rows),
            processing_class=tokenizer,
            callbacks=[Progress(stage, rnd)],
        )
        trainer.train()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        score = heldout_score(stage)
        rnd += 1
        history.append({"stage": stage, "round": rnd, "heldout_solved": score})
        say("STAGE_EVAL " + json.dumps(history[-1]))
        if score >= GATE or rnd >= MAX_ROUNDS:
            say(f"stage {stage + 1} {'passed' if score >= GATE else 'not passed, moving on'} "
                f"({score:.0%} held-out solved)")
            stage, rnd = stage + 1, 0
        push_state()

    say(f"DONE https://huggingface.co/{HUB_RL}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
''')

STAGE_NAMES = ["print short", "print words", "one char", "line loops", "memory cells",
               "digit math", "conditionals", "combinations"]
COLORS = ["#2563EB", "#0D9488", "#D97706", "#7C3AED", "#DB2777", "#059669", "#DC2626", "#4B5563"]
points, evals, log_tail, all_lines = [], [], [], []
t0 = time.time()


def draw():
    clear_output(wait=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    x = list(range(len(points)))
    for s in sorted({p["stage"] for p in points}):
        idx = [i for i, p in enumerate(points) if p["stage"] == s]
        axes[0].scatter(idx, [points[i]["solved"] or 0 for i in idx], s=10, color=COLORS[s],
                        label=f"{s + 1} {STAGE_NAMES[s]}")
        axes[1].scatter(idx, [points[i]["reward"] or 0 for i in idx], s=10, color=COLORS[s])
    if points:
        w = 10
        roll = [sum((p["solved"] or 0) for p in points[max(0, i - w + 1):i + 1]) /
                len(points[max(0, i - w + 1):i + 1]) for i in x]
        axes[0].plot(x, roll, color="black", lw=1.5)
        axes[1].plot(x, [p["zero_std"] or 0 for p in points], color="#9CA3AF", lw=1,
                     label="groups with no reward spread")
    axes[0].set_title("train: fully solved (dots) + rolling mean"); axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_title("reward")
    if points:   # legends only once there is something labelled to show
        axes[0].legend(fontsize=7, loc="upper left")
        axes[1].legend(fontsize=7)
    labels = [f"S{e['stage'] + 1}r{e['round']}" for e in evals]
    axes[2].bar(range(len(evals)), [e["heldout_solved"] for e in evals],
                color=[COLORS[e["stage"]] for e in evals])
    axes[2].axhline(0.6, color="black", ls="--", lw=1)
    axes[2].set_xticks(range(len(evals)), labels, fontsize=7, rotation=45)
    axes[2].set_ylim(0, 1); axes[2].set_title("held-out solved per round (gate 60%)")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6)
    plt.tight_layout()
    plt.show()
    if points:
        p = points[-1]
        per_step = (time.time() - t0) / len(points)
        print(f"stage {p['stage'] + 1}/8 ({STAGE_NAMES[p['stage']]})  round {p['round'] + 1}  "
              f"step {p['step']}/{p['max_steps']}  solved {p['solved'] or 0:.2f}  "
              f"reward {p['reward']:.2f}  len {p['length'] or 0:.0f}  "
              f"{per_step:.0f}s/step  round eta {per_step * (p['max_steps'] - p['step']) / 60:.0f}m  "
              f"elapsed {(time.time() - t0) / 60:.0f}m", flush=True)
    for line in log_tail[-6:]:
        print(line, end="")


with subprocess.Popen(
    ["accelerate", "launch", "--multi_gpu", "--num_processes", "2", SCRIPT],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
) as proc:
    for line in proc.stdout:
        # tqdm leaves "\r" + spaces in front of our tagged lines: match the tag anywhere
        for tag_name in ("PROGRESS ", "STAGE_EVAL "):
            if tag_name in line:
                line = line[line.index(tag_name):]
                break
        if line.startswith("PROGRESS "):
            # newer transformers logs values as strings ('0.1256'); normalize to floats
            p = json.loads(line[9:])
            for k in ("reward", "solved", "pass", "zero_std", "length"):
                try:
                    p[k] = float(p[k]) if p[k] is not None else None
                except (TypeError, ValueError):
                    p[k] = None
            if p["reward"] is None:
                p["reward"] = 0.0
            points.append(p)
            draw()   # a step takes ~70s, so redraw every step
        elif line.startswith("STAGE_EVAL "):
            evals.append(json.loads(line[11:]))
            log_tail.append(line)
            draw()
        elif "it/s]" not in line and "s/it]" not in line:
            log_tail = (log_tail + [line])[-60:]
            all_lines.append(line)
            if not points:
                print(line, end="", flush=True)
    code = proc.wait()

draw()
if code != 0:
    # the launcher's own traceback at the end is noise; show the first real one
    start = next((i for i, l in enumerate(all_lines) if "Traceback" in l),
                 max(len(all_lines) - 40, 0))
    print("\n=== training FAILED, first error ===\n" + "".join(all_lines[start:start + 45]))
else:
    print(f"\nfinished in {(time.time() - t0) / 60:.0f} min")
