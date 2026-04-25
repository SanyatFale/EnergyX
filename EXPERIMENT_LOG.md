# EnergyX LLM Benchmark — Experiment Log

Living document. Update after every run. Follow the entry format strictly so results are comparable across experiments.

---

## Protocol

### System Under Test
Three LLM roles in EnergyX:

| Phase | Component | File |
|---|---|---|
| Routing | `classify_query()` | `energyx/orchestrator/router.py` |
| Tool-calling | `TinyTSAgent.execute()` first call | `tinyts/agent.py` |
| Synthesis | RAG answer generation | `knowledge/rag/synthesizer.py` |

### Model Catalogue

**Active (ready to run)**

| Model ID | Provider | Tier | Context | Notes |
|---|---|---|---|---|
| `cerebras/llama3.1-8b` | Cerebras cloud | small | 8k | Production default |
| `cerebras/gpt-oss-120b` | Cerebras cloud | large | 8k | Primary large-model baseline |
| `ollama/deepseek-r1:1.5b` | Ollama local | small | 4k | Reasoning-distilled; capability floor |
| `ollama/llama3.2:3b` | Ollama local | small | 4k | Compact Meta model; viability test |

**Pending pull (`ollama pull <name>`)**

| Model ID | Pull command | Status |
|---|---|---|
| `ollama/mistral:7b` | `ollama pull mistral:7b` | pending |
| `ollama/qwen2.5:7b` | `ollama pull qwen2.5:7b` | pending |
| `ollama/gemma3:9b` | `ollama pull gemma3:9b` | pending |
| `ollama/phi4:14b` | `ollama pull phi4:14b` | pending |

Once pulled, add them to the active run with:
```bash
python -m benchmark.run_benchmark --models ollama/mistral:7b ollama/qwen2.5:7b ollama/gemma3:9b ollama/phi4:14b
```

### Token Budget (fixed — do not increase without documenting reason)

| Phase | Max tokens / call | Rationale |
|---|---|---|
| Routing | 10 | Single intent word sufficient |
| Tool-call | 200 | Tool name + JSON args for one call |
| Synthesis | 150 | 2-sentence grounded answer |
| Judge | 20 | YES/NO + ≤15-word reason |

### Dataset Summary

| Dataset | Items | Balance |
|---|---|---|
| `routing_test.jsonl` | 75 | 20 analysis, 20 knowledge, 15 control, 15 status, 5 ambiguous |
| `tool_calling_test.jsonl` | 40 | 20 forecast, 10 anomaly, 7 CF-fwd, 3 CF-inv |
| `synthesis_test.jsonl` | 25 | 6 regulations, 5 tariffs, 4 efficiency, 4 smart/grid, 3 MEES/EPC, 3 other |

### Difficulty Distribution

| Dataset | Easy | Medium | Hard |
|---|---|---|---|
| Routing | 43 | 20 | 12 |
| Tool-calling | 17 | 16 | 7 |
| Synthesis | 8 | 12 | 5 |

### Experimental Controls

- Temperature: **0.0** for seed-0 (deterministic baseline), **0.1** for seeds 1–2 (routing/synthesis)
- Seeds: **3** for routing and synthesis; **1** for tool-calling (tool schemas are largely deterministic)
- Prompt template: **fixed** — identical across all models
- Judge model: **cerebras/llama3.3-70b** (strongest available, separate from models under test)
- Statistical tests: McNemar's (binary outcomes), Wilcoxon signed-rank (continuous), Bonferroni correction for >2 model pairs

### Run Command

```bash
# All phases, all models
python -m benchmark.run_benchmark

# Specific models only
python -m benchmark.run_benchmark --models cerebras/llama3.1-8b ollama/mistral:7b

# Single phase
python -m benchmark.run_benchmark --phases routing --seeds 3

# Skip LLM judge (faster, no faithfulness scores)
python -m benchmark.run_benchmark --no-judge

# Re-run stats only on existing results
python -m benchmark.run_benchmark --stats-only
```

---

## Hypotheses

**H1 (Scale):** Larger models (70B+) will outperform small models (7–9B) on routing accuracy by ≥10 percentage points, particularly on hard/ambiguous queries.

**H2 (Tool-calling):** Tool-calling accuracy will be the most differentiating phase — smaller models are expected to call tools in wrong order (e.g., `explain_anomalies` before `detect_anomalies`) more frequently than larger ones.

**H3 (Cloud vs Local):** Cerebras-hosted models will show lower latency (p50 < 500ms) compared to Ollama local models (p50 > 1000ms) for equivalent parameter counts, but accuracy should be equivalent for the same model family.

**H4 (Synthesis Faithfulness):** All models should show high faithfulness scores (>0.85) when context is directly relevant, since the prompt explicitly constrains grounding. Models will diverge most on hard items where the context is incomplete.

**H5 (Reasoning distillation):** DeepSeek R1 1.5B, despite having the fewest parameters, may outperform Llama 3.2 3B on hard tool-calling cases due to its chain-of-thought reasoning distillation, even though routing and synthesis may favour Llama 3.2's broader instruction-following training.

---

## Experiment Entries

### EXPT-001 — Baseline Full Run

**Date:** _to be filled_
**Run by:** _to be filled_
**Command:**
```bash
python -m benchmark.run_benchmark --seeds 3
```

**Models:** All 8 (see catalogue above)

**Results — Routing**

| Model | Acc (mean±std) | Macro F1 | p50 latency |
|---|---|---|---|
| cerebras/gpt-oss-120b | | | |
| cerebras/llama3.1-8b | | | |
| ollama/llama3.2:3b | | | |
| ollama/deepseek-r1:1.5b | | | |

**Results — Tool-calling (First-Tool Accuracy)**

| Model | FTC Acc | Arg Acc | Easy | Medium | Hard |
|---|---|---|---|---|---|
| cerebras/gpt-oss-120b | | | | | |
| cerebras/llama3.1-8b | | | | | |
| ollama/llama3.2:3b | | | | | |
| ollama/deepseek-r1:1.5b | | | | | |

**Results — Synthesis**

| Model | ROUGE-L (±std) | Faithful | Complete |
|---|---|---|---|
| cerebras/gpt-oss-120b | | | |
| cerebras/llama3.1-8b | | | |
| ollama/llama3.2:3b | | | |
| ollama/deepseek-r1:1.5b | | | |

**Statistical Tests**

| Pair | Phase | Test | Statistic | p-value (raw) | p-value (Bonferroni) | Significant? |
|---|---|---|---|---|---|---|
| | | | | | | |

**Hypothesis Assessment**

| Hypothesis | Supported? | Notes |
|---|---|---|
| H1 (Scale → Routing) | | |
| H2 (Tool-calling differentiates) | | |
| H3 (Cloud latency) | | |
| H4 (Synthesis faithfulness high) | | |
| H5 (R1 reasoning distillation) | | |

**Key Findings:**
_To be filled after run._

**Failures / Errors:**
_Document any models that crashed, timed out, or had high error rates._

**Next Steps:**
_To be filled after run._

---

### EXPT-002 — Ablation: Prompt Verbosity

**Planned.** Vary tool schema description verbosity (terse vs. full docstrings) to test whether richer schema descriptions improve first-tool accuracy for smaller models.

**Hypothesis:** Small models (7–9B) will benefit more from verbose tool descriptions than large models, where the gain is expected to be <2pp.

---

### EXPT-003 — Ablation: Temperature Sensitivity

**Planned.** Run routing phase with temperatures [0.0, 0.1, 0.3, 0.7] to quantify variance and confirm that 0.0–0.1 is the right operating range for routing.

---

### EXPT-004 — Error Analysis: Hard Queries Deep-Dive

**Planned.** After EXPT-001, manually inspect all hard-difficulty routing errors. Categorise failures as: (a) keyword confusion, (b) cross-class ambiguity, (c) truncation/refusal. Use findings to update `_ROUTING_PROMPT` if needed.

---

## Error Code Reference

| Code | Meaning |
|---|---|
| `error:llm_failed` | LLM returned None after all retries |
| `error:bind_tools_failed` | Model does not support tool binding |
| `error:timeout` | Call exceeded 30s per-call timeout |
| `pred:unknown` | LLM response did not contain a valid intent word |

---

## Result Files

All results are written to `benchmark/results/` as JSON:

| File | Contents |
|---|---|
| `routing_results.json` | Per-model accuracy, F1, per-item predictions, latency |
| `tool_calling_results.json` | FTC accuracy, arg accuracy, by-difficulty breakdown |
| `synthesis_results.json` | ROUGE-L, faithfulness, completeness, per-item responses |

---

## Glossary

| Term | Definition |
|---|---|
| **FTC Acc** | First-Tool-Call Accuracy: fraction of cases where the model's first tool call matches the gold standard |
| **Macro F1** | Unweighted average F1 across all intent classes |
| **Cohen's κ** | Inter-annotator agreement (target: κ > 0.8 for publication) |
| **ROUGE-L** | Recall-oriented understudy for gisting evaluation using longest common subsequence |
| **Faithfulness** | Judge YES/NO: response does not hallucinate facts beyond the provided context |
| **Bonferroni** | Correction dividing α by number of comparisons to control family-wise error rate |
| **TCR** | Task Completion Rate: agent reached a terminal state with valid answer |
| **p50 / p95** | 50th / 95th percentile of latency distribution |
