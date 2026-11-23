# Mathematics in Modern LLMs Companion Code

SymPy starters, exact calculations, and executable references for the first
edition of *Mathematics in Modern LLMs* by Yin Yang, published by the Foundation
Books Project. The book supplies the assignment inputs and includes worked
answers to the written exercises. Use the chapter guide to find the student
starter, then consult the executable reference after attempting the homework.
Appendix A includes optional evaluation calculations.

## November 2026 first edition

Use the [fixed first-edition snapshot](https://github.com/foundation-books/mathematics-in-modern-llms-code/tree/60aed03f49735f6d9baefaf067b778e684c757d3)
to download the files. In a clone of this repository, run
`git checkout --detach 60aed03f49735f6d9baefaf067b778e684c757d3` before setup.

## Setup

Python 3.10 is the minimum. Run the commands in this guide and in the chapter
guides from the companion repository root. In the manuscript checkout, enter
the companion directory with `cd code` first.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
python3 check.py
```

The first-edition checks used Python 3.12.3 with SymPy 1.13.3, mpmath 1.3.0,
and pytest 8.3.4.
Use `requirements.txt` if you only need to run examples. The examples use
small CPU fixtures and run offline after setup. Keep Python assertions
enabled when running the checks.

To start Chapter 1, run `python3 chapter_language_modeling/symbolic_warmup.py`.
It prints `Matrix([[-1/4], [1/4]])`. The chapter guides identify the assignment
inputs and starter files, optional helpers and extensions, and completed
references.

## Run the checks

```sh
python3 check.py --list
python3 check.py --chapter 6
python3 check.py --chapter 2 13 20
python3 check.py --appendix A
python3 check.py --report /tmp/mml-companion-checks.json
```

The default run checks Chapters 1–20 and Appendix A. Each starter, reference,
and test suite runs in a fresh process. A failing command makes the checker
exit with an error. The optional JSON report includes versions, exit statuses,
and program output. In a standalone checkout, only tests that explicitly
compare against manuscript source are skipped.

`chapters.json` lists the executable entry points in book order. Some chapters
also contain notebooks or calculations from earlier assignments. The chapter
guides explain their role and the current printed homework defines the task.

## Chapter index

| Chapter | Companion and guide | Main checks |
| --- | --- | --- |
| 1 | [Language modeling](chapter_language_modeling/README.md) | Sequence probabilities, loss, entropy/KL, masks and prefix codes |
| 2 | [Matrix derivatives and gradient flow](chapter_differentiation/README.md) | Independent matrix gradients, exact fixtures and frozen-target checks; optional Jacobian and precision extensions |
| 3 | [Tokens and embeddings](chapter_tokens_embeddings/README.md) | Merge counts, lookup gradients, rotary identities and causal masking |
| 4 | [Softmax attention](chapter_softmax_attention/README.md) | Forward/backward attention, masks, shared-input gradients and stability |
| 5 | [Efficient attention](chapter_efficient_execution/README.md) | Online summaries, causal tiles, reconstructed gradients, memory traffic and rounded reductions |
| 6 | [Attention state](chapter_attention_state/README.md) | MHA/GQA/MQA, latent absorption, partial RoPE and cache accounting |
| 7 | [Sparse and hierarchical attention](chapter_sparse_hierarchical_attention/README.md) | Restricted support, causal routing, pooling, hierarchy and exact pair counts |
| 8 | [Linear attention and SSMs](chapter_linear_attention_ssm/README.md) | Pairwise/recurrent feature attention, state-space expansion and affine scans |
| 9 | [Delta memory](chapter_delta_memory/README.md) | Corrective gradients, gates, sequential/affine/chunked updates and reset cases |
| 10 | [Gates and normalization](chapter_gates_normalization/README.md) | Normalization Jacobians, null directions, gate paths and central differences |
| 11 | [Learning across depth](chapter_learning_depth/README.md) | Residual reads, stream mixing, dynamic derivatives and finite Sinkhorn sweeps |
| 12 | [MoE routing](chapter_moe_routing/README.md) | Sparse dispatch, balanced placement, remote-type bounds, output/gradient equivalence and accounting |
| 13 | [Conditional memory](chapter_conditional_memory/README.md) | Hash addresses, streaming suffixes, fusion Jacobians, scatter-add and collisions |
| 14 | [Optimization](chapter_optimization/README.md) | AdamW, supported polar factors, sharing-aware QK clipping, feedback timing and optimizer ownership |
| 15 | [Scaling and allocation](chapter_scaling_allocation/README.md) | Constructed fits, constrained optima, acquisition laws, importance variance and adaptive collection |
| 16 | [Low precision](chapter_low_precision/README.md) | Exact quantizers, rounding ties, reconstruction errors, surrogate gradients and storage |
| 17 | [Post-training](chapter_adaptation_transfer/README.md) | Low-rank updates, visited histories, regularized preferences and complete-response DPO |
| 18 | [Reinforcement learning](chapter_reinforcement_learning/README.md) | Policy gradients, all 256 groups, GAE, localized policy movement and joint rollout-admission correction |
| 19 | [Speculative decoding](chapter_speculative_decoding/README.md) | Categorical correction, exact block/EOS laws, MTP losses, byte-language constraints and synchronized rollback |
| 20 | [Discrete diffusion](chapter_discrete_diffusion/README.md) | Corruption kernels, Bayes reversal, joint dependence and synchronous counterexamples |
| A | [Model evaluation](appendix_model_evaluation/README.md) | Paired variance/covariance, exact binary tests, document means, factorial effects, mixtures and constrained choices |

## License

See [LICENSE](LICENSE) for the original companion code and documentation and
[NOTICE.md](NOTICE.md) for the scope of that license.
