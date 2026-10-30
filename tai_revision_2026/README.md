# Reward search for LLM reasoning: revised study (2026)

This folder holds the code and results of the revised study of reward function search for GRPO
(six searches, 259 screened rewards, 468 trained policies, held out evaluation on seven mathematics
benchmarks plus GPQA-Diamond and MedQA). The files at the top of this repository belong to the first
version of the work (GSM8K only).

## Layout

| Folder | Content |
|---|---|
| `code/` | Training, search, evaluation, statistics and figure data scripts |
| `runs/<search>/<policy>/rounds/` | Every generated reward per round (`roundN_rewards.json`: name, description, Python source), the feedback shown to the generator and the ranking after each round |
| `runs/<search>/<policy>/results/` | Validation and test results of every trained policy |
| `runs/<search>/<policy>/bench/` | Held out benchmark summaries per policy (per question records are in the release archive) |
| `runs/<search>/<policy>/outputs/*/train_meta.json` | Training configuration and provenance of each run |
| `results/cmp/` | Paired comparisons (bootstrap intervals, sign tests) behind every difference in the paper |
| `results/paper/` | Paper numbers, figure data and the compute total |
| `results/pools/` | Training free analysis of every reward on sampled answers |
| `results/decontamination_*.json` | Contamination audit and the list of removed training rows |
| `paper_tools/` | Scripts that draw the figures, and the Fig. 1 source |

Search folders: `gsm8k` (Llama-3.2-3B-Instruct, ranked feedback), `math` (ranked), `math_fa` (failure
feedback), `math_v4` (literature seeded search with the diversity filter and seed confirmation),
`math_lit` (published reward baselines and controls on the filtered MATH split), `dapo`, `medmcqa`, `math_xmodel` (published rewards on three more policies), `math_lowvar`
(lower variance screening pilot).

## Main entry points

- `code/orchestrator_v3.py`: one search (generate, sandbox, screen with GRPO, rank, feed back).
- `code/train.py`: one GRPO run (TRL, vLLM rollouts, LoRA) with the base rewards plus candidates.
- `code/common.py`: base rewards, task verifiers, sandbox, data splits.
- `code/literature_rewards.py`: the eleven published rewards used as baselines.
- `code/eval_benchmarks.py`: the held out suite.
- `code/bench_compare_seeds.py` and `code/cmp_paper.sh`: paired comparisons pooled over seeds.
- `code/paper_numbers.py`, `code/fig_data.py`, `code/compute_total.py`: every number and figure in the paper.
- `code/sample_pool.py`, `code/score_rewards.py`, `code/analyze_pool.py`: training free reward analysis.
- `code/decontaminate.py`: the 13 word overlap audit.

## Release archive

The GitHub release `tai-revision-2026` attaches `records_logs_rollouts.tar.gz` (about 0.5 GB): per
question benchmark records, evaluation logs, training logs, reward traces and rollouts of every run,
plus the sampled answer pools used by the training free analysis and the generator log. LoRA adapters
(about 170 GB) are not included; they are available from the authors on request.

## License

The licenses at the top of this repository apply to everything in this folder.
