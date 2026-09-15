# bf-qwen-brainfuck

Teaching Qwen2.5-Coder-1.5B to write Brainfuck, with supervised fine-tuning first and then reinforcement learning against a Brainfuck interpreter.

The final model is on Hugging Face as [hd2514p/bf-qwen-1.5b-grpo-v8b](https://huggingface.co/hd2514p/bf-qwen-1.5b-grpo-v8b). [MODEL_CARD.md](MODEL_CARD.md) has the full write-up.

## Results

| Evaluation | Base Qwen 1.5B | This model |
|---|---|---|
| BF-Ladder, my frozen test (100 problems) | 0 / 100 | 21 / 100 |
| EsolangBench official harness, zero-shot | not run | 2 / 80 (2.5%) |
| EsolangBench official harness, self-scaffolding | not run | 2 / 80 (2.5%) |

On zero-shot Brainfuck, 2.5% matches GPT-5.4, O4-mini, Gemini 3.1 Pro and Qwen-235B in the [EsolangBench paper](https://arxiv.org/abs/2603.09678). It's a tie at 2 problems, and this model was trained on Brainfuck while those weren't, so read it as a small result. The model card goes through the caveats.

Every number comes from problems, wording and parameter values that never appeared in training.

## Built on bf-gym

All of this runs on [bf-gym](https://github.com/hsd2514/bf-gym), my Brainfuck environment. Every script uses its interpreter (`bf_gym.interpreter.run_bf`) to check training programs, compute RL rewards and grade evaluations. The strict mode (`BFConfig(wrap_pointer=False)`) matches the official grader, so moving the pointer left of cell 0 is an error. The first training stage also uses bf-gym's task families (`bf_gym.tasks`) to generate problems.

```
pip install --no-deps git+https://github.com/hsd2514/bf-gym.git
```

## How the model answers

The model writes a short plan, then the program:

```
PLAN
cell 0 stays 0 as an end marker
read chars into cells 1 2 3 and so on until input ends
step back to the last char
print and move left until reaching the marker
CODE
>,[>,]<[.<]
```

Only the part after `CODE` gets run. The plan contains `.`, `,` and `-`, which are Brainfuck commands, so running the whole answer would corrupt the program.

## Repository layout

```
training/
  1_sft_plan_then_code.py      SFT from the base model: plan + code pairs
  2_sft_character_codes.py     SFT: plans that state character codes and arithmetic
  3_grpo_curriculum.py         GRPO with an 8-stage curriculum, interpreter as reward
evaluation/
  bf_ladder/
    build_ladder.py            builds the frozen 100-problem test (run once)
    bf_ladder_v1.json          the frozen test, with verified reference solutions
    eval_ladder.py             scores models on BF-Ladder
  esolangbench_official.py     runs the official EsolangBench harness on the model
  compare_checkpoints.py       compares checkpoints on held-out tasks and the benchmark
  diagnose_answers.py          prints plans, code and interpreter output per task
MODEL_CARD.md                  the Hugging Face model card
```

Each script is a single notebook cell. Stage 1 was written for Colab and everything else for Kaggle with two T4 GPUs. Paste a script into one cell and run it. The cells install their own dependencies and read a Hugging Face token from the notebook's secrets (`HF_TOKEN`).

## Training pipeline

Run the three training scripts in order. Each one starts from the adapter the previous one pushed.

**1. Plan then code (SFT).** Verified plan and code pairs, generated from bf-gym task families and checked with the strict interpreter. Prompts mix the benchmark's format, bf-gym's format and a neutral wording, so the model doesn't learn a single template.

**2. Character codes (SFT).** The first model wrote plans like "7 x 12 = 84 makes 'j'", right in shape but with wrong numbers, because 'j' is 106. This stage trains on plans that state each character's code and the arithmetic to reach it, over random strings. Printing unseen text went from 0% to 92%.

**3. GRPO curriculum.** The model writes programs, the interpreter runs them on freshly generated hidden tests, and the reward is the fraction of tests passed. There are 8 stages, one concept each. The model moves on after solving 60% of variants with parameter values it never trained on. A stage it already passes is skipped. The adapter is pushed to the Hub after every round, so a stopped Kaggle session picks up where it left off.

The Hugging Face repos for the two SFT stages were deleted after training. Running the scripts in order recreates them.

## Evaluation

**BF-Ladder** is my own test. 10 skill levels of 10 problems. Levels 1 and 2 print fixed text, so they take no input and have one test each. The other problems have 5 test cases. Levels 1 to 6 cover what training taught, and levels 7 to 10 (if/else, count and filter, multi-digit numbers, comparisons) were never trained. Every problem ships with a reference solution, and the ones that read input are also verified on 200 extra random inputs. For those, the expected outputs differ between test cases, so a program that ignores its input can't pass. The file is frozen. Don't edit it after scoring models and don't train on it. If the test needs changes, make a `v2`.

**EsolangBench** runs the unmodified official harness against a local model server. There are two patches, both applied at runtime. The API URL points at the local server, and a 200,000-step limit stops programs that never halt from stalling the run.

## What I learned

- RL can only reinforce something the model sometimes gets right. The first GRPO run started from a model that didn't know character codes. It earned zero reward for three rounds and learned nothing.
- SFT with explicit reasoning in the plan fixed missing knowledge that RL couldn't discover on its own.
- RL gains stayed inside the task generator it trained on. Printing unseen words improved from 12% to 46% on curriculum tasks and didn't move on BF-Ladder.
- Check the evaluation before trusting it. Along the way I found a model server that ignored temperature, answers cut off by a token limit, and a test that a program printing a constant "yes" could pass.

## Limitations

The model can't write conditionals. If/else, counting, multi-digit numbers and comparisons all score 0, and most EsolangBench problems need at least one of them. Small changes in wording or numbers also break it.
