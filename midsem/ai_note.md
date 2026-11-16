I used an AI coding assistant (Claude Code, Anthropic) extensively for this assignment. It wrote:
- **Part A code**: the LoRA module (`partA/lora.py`), the DPO loss and training loop (`partA/train_dpo.py`), the LoRA sweep (`partA/train_lora.py`), the test-set and preference-pair generation (`partA/build_data.py`) and shared helpers (`common.py`). These reuse my own project model (`phase2/model.py`), tokenizer, data generator and Phase 3 data format unchanged.
- **Part C code**: the answer checker, the 20 Hindi problems, and the C1 and C2 scripts.
- **Part B**: the worked calculations and first drafts of the written answers.
- **Report**: the first draft of this report and the figure scripts.

All experiments were run on my machine and every number is taken from the logged JSON results in `midsem/results/`.

**[TODO before submitting — confirm this yourself, or edit it:]** I have read and checked every line of code and every number submitted, and can explain them.
