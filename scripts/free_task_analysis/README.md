# Why episodes fail on the free-choice tasks

Three small scripts that read the logged calls and true block positions of a finished benchmark
run (camera right). They produced the numbers on where the block-to-role assignment is held
(A2: prompt history, A9: fixed in code, A10: in code and revisable).

```bash
# from the repository root; BENCH names one model's benchmark folder, the arguments its agent folders
python scripts/free_task_analysis/free_tax.py A2 A9 A10r-none A10-none      # failure categories per family
python scripts/free_task_analysis/free_why.py A2 A9 A10-none                # re-picking placed blocks, retries, spare blocks
python scripts/free_task_analysis/free_parse.py A9 A10r-none                # how the open instruction was read into goals
BENCH=results/bench/qwen3.8_27b python scripts/free_task_analysis/free_tax.py A2__free A9__free A10-none__free
```
