---
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- brainfuck
- esolang
- code-generation
- lora
- grpo
- reinforcement-learning
language:
- en
---

# bf-qwen-1.5b-grpo-v8b

A LoRA adapter for Qwen2.5-Coder-1.5B-Instruct that writes Brainfuck programs. The model first writes a short plan (what each cell holds, which character codes it needs) and then the program.

The base model can't write working Brainfuck. It solves 0 of the 100 problems in my test set. With this adapter it solves 21. Every score below comes from problems, wording and parameter values the model never saw in training.

## How it was trained

Training ran in two phases on Kaggle T4 GPUs.

**Supervised fine-tuning.** The adapter learned from programs that a strict Brainfuck interpreter had checked. Each example pairs a plan with its code, and the plan is built from the same numbers as the program, so the two always agree. Early versions memorized about 35 program templates and copied the nearest one for any new task. Two changes fixed most of that. Tasks got random strings and parameters, so no fixed answer works. And the plans started stating character codes and arithmetic explicitly ("'j' is 106. 106 = 10 x 10 + 6"). That second change took printing unseen text from 0% to 92%.

**Reinforcement learning (GRPO).** The model never sees a solution in this phase. It writes a program, the interpreter runs it on freshly generated hidden tests, and the reward is the fraction of tests passed. A small penalty applies to answers without the plan/code format or with unbalanced brackets. There is no partial credit for output that merely looks close. Training follows a curriculum of 8 stages, one concept each. The model moves to the next stage after solving 60% of held-out variants, meaning tasks with constants, letters or separators it never trained on.

LoRA settings: rank 16, alpha 32, dropout 0.05, applied to all attention and MLP projections.

The Brainfuck interpreter, the task families for the first training stage and the grading all come from [bf-gym](https://github.com/hsd2514/bf-gym), my Brainfuck environment. Its strict mode treats moving the pointer left of cell 0 as an error, the same rule the official EsolangBench grader uses.

## Results

Nothing in these evaluations was used for training.

### BF-Ladder (my test set, frozen before evaluation)

100 problems across 10 skill levels. Problems that read input have 5 test cases, and the fixed-text problems in levels 1 and 2 have one. Grading is strict. Every test must pass, and moving the pointer left of cell 0 counts as an error. Every problem has a reference solution that passes its tests, and the ones that read input are also verified on 200 random inputs. The wording ("Write a Brainfuck program that does the following") and the parameter values differ from anything in training.

| Level | Base model | This model |
|---|---|---|
| 1. Print a character | 0 | 10 |
| 2. Print text | 0 | 5 |
| 3. One character in | 0 | 2 |
| 4. Loop over input | 0 | 0 |
| 5. Several cells | 0 | 1 |
| 6. Digit math | 0 | 3 |
| 7. If / else | 0 | 0 |
| 8. Count and filter | 0 | 0 |
| 9. Multi-digit numbers | 0 | 0 |
| 10. Comparisons | 0 | 0 |
| **Total (of 100)** | **0** | **21** |

### EsolangBench (official harness)

Scored with the unmodified [EsolangBench](https://github.com/Lossfunk/EsolangBench) harness on all 80 Brainfuck problems. The harness gets only the code after the plan. I added a 200,000-step limit to its interpreter so that programs that never halt can't stall the run.

| Regime | Solved |
|---|---|
| zero_shot | 2 / 80 (2.5%) |
| self_scaffolding (5 attempts with interpreter feedback) | 2 / 80 (2.5%) |

The two solved problems are echo and reverse. Asking the same 80 problems in wording never used in training also gives 2 of 80.

### Compared with the EsolangBench paper

The [EsolangBench paper](https://arxiv.org/abs/2603.09678) (Table 1) reports these Brainfuck scores on the same 80 problems:

| Model | Zero-shot | 3-shot |
|---|---|---|
| GPT-5.4 xhigh | 2.5% | 2.5% |
| O4-mini | 2.5% | 3.8% |
| Gemini 3.1 Pro | 2.5% | 3.8% |
| Qwen-235B | 2.5% | 1.2% |
| Kimi K2.5 | 0% | 0% |
| **This model (1.5B)** | **2.5%** | **not tested** |

On zero-shot Brainfuck this 1.5B adapter matches the frontier models. Like every model in the paper, it solves nothing beyond the Easy tier.

Read this as a tie at the smallest possible margin, not a win. 2.5% is 2 problems, and one problem moves the score by 1.25%. The comparison is also uneven. The paper's models had no Brainfuck-specific training. This model was fine-tuned on Brainfuck, and echo and reverse tasks were part of its training data, though worded differently from the benchmark. Some training prompts also used the benchmark's prompt format. The unseen-wording run still scores 2 of 80, so the result doesn't depend on that format. I haven't run the 3-shot setting, so the best 3-shot score of 3.8% has no counterpart here.

### Held-out curriculum tasks

On the 8 curriculum stages with held-out parameter values (24 tasks per stage), this model solves 35%. The supervised checkpoint solves 29%. The biggest change was printing unseen words, which went from 12% to 46%.

## Limitations

- It can't write conditionals. If/else, counting matches, multi-digit numbers and comparisons all score 0. Most EsolangBench problems need at least one of these, which is why the benchmark score is low.
- Small changes in wording or numbers can break it. It trained on "shift a letter by 1 to 5" and still solves only 2 of 10 problems that ask for a shift of 6 or 7.
- RL improved the kinds of tasks it practiced on but didn't carry over to new phrasings. The gain shows on the curriculum tasks and disappears on BF-Ladder.
- Long constants sometimes come out one or two `+` short even when the plan's arithmetic is right.

## Usage

The model answers with `PLAN`, a few lines, `CODE`, then the program. Run only the part after `CODE`, because periods, commas and hyphens in the plan are Brainfuck commands.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, "hd2514p/bf-qwen-1.5b-grpo-v8b")

messages = [{"role": "user", "content": "Write a Brainfuck program that reads a line and prints it backwards."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=768, do_sample=False)
text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

code = text.rsplit("\nCODE\n", 1)[-1].strip() if "\nCODE\n" in text else text.strip()
print(text)
print("program:", code)
```

An answer has this shape (the program is the one the model wrote for the benchmark's reverse problem):

```
PLAN
cell 0 stays 0 as an end marker
read chars into cells 1 2 3 and so on until input ends
step back to the last char
print and move left until reaching the marker
CODE
>,[>,]<[.<]
```
