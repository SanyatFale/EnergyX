"""Metric implementations for the EnergyX LLM benchmark.

Three evaluation layers:
  1. Routing   — classification accuracy, macro F1, Cohen's κ, latency
  2. Tool-call — Tool F1 (ToolBench-style), step efficiency, arg accuracy
  3. Synthesis — ROUGE-L, faithfulness (via judge YES/NO), citation accuracy

Statistical tests:
  - McNemar's test   (paired binary outcomes between two models)
  - Wilcoxon signed-rank (continuous metrics, non-parametric)
  - Bonferroni correction (family-wise error rate for >2 model comparisons)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Routing metrics
# ---------------------------------------------------------------------------

def routing_accuracy(preds: List[str], labels: List[str]) -> float:
    assert len(preds) == len(labels)
    return sum(p == l for p, l in zip(preds, labels)) / len(labels)


def macro_f1(preds: List[str], labels: List[str]) -> Dict[str, float]:
    """Returns per-class P/R/F1 and macro average."""
    classes = sorted(set(labels))
    results: Dict[str, Any] = {}
    f1s = []
    for cls in classes:
        tp = sum(p == cls and l == cls for p, l in zip(preds, labels))
        fp = sum(p == cls and l != cls for p, l in zip(preds, labels))
        fn = sum(p != cls and l == cls for p, l in zip(preds, labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[cls] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        f1s.append(f1)
    results["macro_f1"] = round(sum(f1s) / len(f1s), 4)
    return results


def cohens_kappa(rater1: List[str], rater2: List[str]) -> float:
    """Compute Cohen's κ between two annotation sequences."""
    assert len(rater1) == len(rater2)
    n = len(rater1)
    classes = sorted(set(rater1) | set(rater2))
    p_o = sum(a == b for a, b in zip(rater1, rater2)) / n
    c1 = Counter(rater1)
    c2 = Counter(rater2)
    p_e = sum((c1[c] / n) * (c2[c] / n) for c in classes)
    return round((p_o - p_e) / (1 - p_e), 4) if (1 - p_e) > 1e-9 else 1.0


def latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)
    return {
        "p50_ms": round(sorted_l[int(n * 0.50)], 1),
        "p95_ms": round(sorted_l[int(n * 0.95)], 1),
        "mean_ms": round(sum(sorted_l) / n, 1),
    }


# ---------------------------------------------------------------------------
# Tool-calling metrics (ToolBench / AgentBench conventions)
# ---------------------------------------------------------------------------

def tool_f1(pred_tools: List[str], gold_tools: List[str]) -> Dict[str, float]:
    """Compute precision, recall, F1 over predicted vs gold tool sets.

    Uses multi-set matching: duplicate calls count.
    """
    pred_c = Counter(pred_tools)
    gold_c = Counter(gold_tools)
    tp = sum(min(pred_c[t], gold_c[t]) for t in gold_c)
    prec = tp / sum(pred_c.values()) if pred_c else 0.0
    rec  = tp / sum(gold_c.values()) if gold_c else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def step_efficiency(pred_seq: List[str], gold_seq: List[str]) -> float:
    """Ratio of gold sequence length to predicted (<=1 means over-calling)."""
    if not pred_seq:
        return 0.0
    return round(len(gold_seq) / len(pred_seq), 4)


def first_tool_correct(pred_tool: Optional[str], expected_tool: str) -> bool:
    return pred_tool == expected_tool if pred_tool else False


def arg_accuracy(pred_args: Dict[str, Any], gold_args: Dict[str, Any]) -> float:
    """Fraction of gold arg keys whose values match in pred_args.

    Numeric args: exact match. String args: case-insensitive contains.
    """
    if not gold_args:
        return 1.0
    hits = 0
    for k, v in gold_args.items():
        if k not in pred_args:
            continue
        pv = pred_args[k]
        if isinstance(v, (int, float)):
            hits += int(pv == v)
        else:
            hits += int(str(v).lower() in str(pv).lower() or str(pv).lower() in str(v).lower())
    return round(hits / len(gold_args), 4)


def hallucination_rate(pred_tools: List[str], valid_tools: List[str]) -> float:
    valid = set(valid_tools)
    if not pred_tools:
        return 0.0
    return round(sum(t not in valid for t in pred_tools) / len(pred_tools), 4)


# ---------------------------------------------------------------------------
# Synthesis metrics
# ---------------------------------------------------------------------------

def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 (LCS-based), no external library dependency."""
    def lcs_len(a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(2)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
                else:
                    dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
        return dp[m % 2][n]

    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()
    if not h_tokens or not r_tokens:
        return 0.0
    lcs = lcs_len(h_tokens, r_tokens)
    prec = lcs / len(h_tokens)
    rec  = lcs / len(r_tokens)
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(f1, 4)


def citation_present(response: str, source_markers: List[str]) -> bool:
    """Check whether the response references any source from the context."""
    resp_lower = response.lower()
    return any(m.lower() in resp_lower for m in source_markers)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def mcnemars_test(
    correct_a: List[bool], correct_b: List[bool]
) -> Dict[str, float]:
    """McNemar's test for two paired binary outcome sequences.

    Null hypothesis: both models make the same type-I/II errors.
    Returns chi2 statistic and two-tailed p-value (chi2 approximation).
    """
    assert len(correct_a) == len(correct_b)
    # b: A correct, B wrong; c: A wrong, B correct
    b = sum(1 for a, x in zip(correct_a, correct_b) if a and not x)
    c = sum(1 for a, x in zip(correct_a, correct_b) if not a and x)
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c}
    # With Yates continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = _chi2_sf(chi2, df=1)
    return {"chi2": round(chi2, 4), "p_value": round(p, 6), "b": b, "c": c}


def wilcoxon_signed_rank(scores_a: List[float], scores_b: List[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank test (exact for small n, normal approx for n>25).

    Tests whether two paired samples come from the same distribution.
    """
    assert len(scores_a) == len(scores_b)
    diffs = [a - b for a, b in zip(scores_a, scores_b) if a != b]
    if not diffs:
        return {"W": 0.0, "p_value": 1.0, "n": 0}
    abs_diffs = sorted(enumerate(abs(d) for d in diffs), key=lambda x: x[1])
    ranks = [0.0] * len(abs_diffs)
    i = 0
    while i < len(abs_diffs):
        j = i
        while j < len(abs_diffs) and abs_diffs[j][1] == abs_diffs[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[abs_diffs[k][0]] = avg_rank
        i = j
    W_plus  = sum(r for d, r in zip(diffs, ranks) if d > 0)
    W_minus = sum(r for d, r in zip(diffs, ranks) if d < 0)
    W = min(W_plus, W_minus)
    n = len(diffs)
    if n > 25:
        mean_W = n * (n + 1) / 4.0
        std_W  = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (W - mean_W) / std_W if std_W > 0 else 0.0
        p = 2 * _norm_sf(abs(z))
    else:
        p = float("nan")  # exact table needed; use scipy for small n
    return {"W": round(W, 2), "p_value": round(p, 6) if not math.isnan(p) else "exact_needed", "n": n}


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """Apply Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    return [min(1.0, round(p * n, 6)) for p in p_values]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function of chi2(df=1): P(X > x). Uses regularised gamma approx."""
    return _gammaincc(df / 2.0, x / 2.0)


def _gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) via series + continued fraction."""
    if x < 0:
        return 1.0
    if x == 0:
        return 1.0
    if x < a + 1:
        return 1.0 - _gammaincl_series(a, x)
    return _gammaincl_cf(a, x)


def _gammaincl_series(a: float, x: float) -> float:
    ap = a
    delta = s = 1.0 / a
    for _ in range(200):
        ap += 1
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * 1e-9:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammaincl_cf(a: float, x: float) -> float:
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, 201):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-9:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _norm_sf(z: float) -> float:
    """P(Z > z) for standard normal using erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2))
