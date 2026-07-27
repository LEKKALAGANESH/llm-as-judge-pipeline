<div align="center">

# llm-as-judge-pipeline

**An LLM-as-judge evaluation pipeline that takes judge bias seriously — naming it, mitigating it in code, and *measuring* it.**

[![tests](https://img.shields.io/badge/tests-200%20passed-brightgreen)](#testing)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![ruff](https://img.shields.io/badge/ruff-clean-brightgreen)](pyproject.toml)
[![offline](https://img.shields.io/badge/tests-no%20API%20key%20needed-informational)](#testing)
[![replay](https://img.shields.io/badge/audit%20trail-replay%20verified-success)](#audit-trail)

</div>

---

Using a strong LLM to score model outputs scales quality assessment past what human review can
reach. It also imports the judge's biases wholesale: preferring the first option, the longer
answer, its own model family, or anything phrased confidently.

Most pipelines *describe* those biases. This one **measures** them — and the distinction is the
whole point.

> **A mitigation with no number beside it is a claim, not a result.**
> The rubric line for bias handling is a three-way conjunction: identified **and** code-mitigated
> **and** empirically measured. This README marks which of the three each bias has.

---

## Table of contents

- [Results](#results)
- [Quickstart](#quickstart)
- [The five biases](#the-five-biases)
- [Judging design](#judging-design)
- [Statistics](#statistics)
- [Audit trail](#audit-trail)
- [Deployment gating](#deployment-gating)
- [Testing](#testing)
- [Limitations](#limitations)

---

## Results

From the committed run in [`results/`](results/). Judge: `groq/llama-3.3-70b-versatile` at T=0.0.

### Suite

| Metric | Value |
|---|---:|
| Cases scored | **18 / 18** (0 parse errors, 0 call failures) |
| Pass rate | **50.0%** — overall ≥ 4 **and** every criterion ≥ 3 |
| Mean overall | **3.33** |
| Score spread | sd **1.680** · entropy **1.927 bits** · modal share 44% |
| Length vs score | ρ = **+0.309** (p=0.212, n=18) |
| Tokens | 73,214 in / 6,305 out · $0.048 paid-tier equivalent |

Per-criterion means: correctness 3.67 · tone 3.67 · completeness 3.50 · faithfulness 3.39 ·
instruction-following 3.28 · safety 4.67.

> **Reading ρ correctly.** +0.309 is *not* a bias estimate on its own. On this suite, length and
> **human-judged** quality genuinely correlate at **ρ = +0.537**. A perfectly calibrated judge
> should land near +0.54, not 0. The corrected estimate is `ρ_judge − ρ_gold`, which is
> **negative** here — the anti-verbosity clause is, if anything, slightly over-correcting.

### Position bias

| Metric | Value |
|---|---:|
| **Flip rate** | **0.0** |
| Pairs resolved in both orders | 3 of 15 |
| Declared winner | `prompt_v2_direct` |

⚠️ **That result would now be blocked by this pipeline's own coverage gate.** Three of fifteen
completed pairs is below the 50% minimum. The flip rate is computed over *completed* pairs, so it
reads a clean 0/3 and would otherwise sail through the validity gate — the failure mode that
actually occurred is precisely the one the old gate could not see. Re-run `compare-configs` to
completion before quoting the winner.

### Cross-family invariant — enforced at startup

```
judge = meta          (llama-3.3-70b-versatile)
candidates = openai   (gpt-oss-120b, gpt-oss-20b)
```

Both run on Groq. The invariant is about model **lineage**, not provider — `derive_family` reads
the model portion, never the routing prefix, so `groq/gemini-3.5-flash` resolves to `google` and
cannot pass for the wrong reason.

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env                 # two keys; see the table below

python main.py smoke-test            # 2 calls, verifies both providers before you spend
python main.py run-suite data/test_suites/general_qa.json
```

| Command | Produces |
|---|---|
| `run-suite` | Pass rate, per-criterion means, score distribution, length correlation |
| `compare-configs` | Win/loss/tie, **flip rate**, Clopper-Pearson CI, declared winner |
| `validate-judge` | Quadratic-weighted κ with bootstrap CI, test-retest ICC, probe pass rates |
| `run-ablation` | Before/after per bias, with paired tests |
| `measure-self-enhancement` | Cross-family vs same-family win-rate delta |
| `gate` | Deployment decision; **exits 1** so CI can use it |
| `replay` | Rebuilds a committed report from the audit log alone |

### What needs an API key

| Runs with **no key at all** | Needs a key |
|---|---|
| **All 200 tests** | Any live judging |
| Every aggregation and statistic | |
| `replay` — rebuilds reports from the log | |

---

## The five biases

Each is marked for all three requirements: **identified**, **mitigated in code**, **measured**.

### 1. Position bias

- **Mitigation** — every pair judged in both orders; disagreement resolves to
  `Tie (Position Inconsistency)` rather than picking a side.
- **Measured** — flip rate **0.0**, plus a 5-probe noise floor using byte-identical outputs.

The remap that maps the reversed winner back is an **involution**, so applying it twice silently
*inverts* the consistency verdict while looking correct. There is exactly one call site
repo-wide, and a test asserts it.

The noise-floor probes set `bypass_cache=True`. Without it, identical outputs produce a
byte-identical prompt, the second call is a guaranteed cache hit, and the control reports one
observation as two — manufacturing a position flip that never happened.

### 2. Verbosity bias

- **Mitigation** — anti-verbosity clause in the rubric; `completeness` narrowed so thoroughness
  is not double-counted.
- **Measured** — ρ = +0.309 against a **gold baseline of +0.537**, plus 6 padded-answer probes.

Correcting against the gold baseline costs zero API calls: both vectors are already on disk.

### 3. Self-enhancement bias

- **Mitigation** — cross-family invariant enforced as a **startup assertion**, derived from the
  model portion rather than the routing prefix. The vacuous-truth hole (an empty candidate list
  makes `all(...)` trivially true) is explicitly closed, and unknown families are refused.
- **Measured** — `measure-self-enhancement` runs the same suite under a deliberately same-family
  judge and reports the win-rate delta with an exact McNemar test.

### 4. Sycophancy / style bias

- **Mitigation** — rationale-before-score enforced **structurally**, through schema field order
  propagated into the wire schema from `model_fields`, so prompt, schema and parser cannot drift
  apart. Asking a model nicely to reason first is not a mitigation; making score-first
  unparseable is.
- **Measured** — 6 confidently-wrong probes with terse-correct controls.

### 5. Score clustering

- **Mitigation** — few-shot 1/3/5 anchors on all six criteria, loaded from versioned YAML.
- **Measured** — standard deviation **1.680**, Shannon entropy **1.927 bits**, modal share 44%,
  top-two share 50%, plus a full histogram.

"We added anchors" is not a measurement. "Scores collapsed to 4-or-5 on X% of cases without
anchors and Y% with" is.

---

## Judging design

Six criteria in [`config/rubric.yaml`](config/rubric.yaml), each with a description, **1/3/5
anchors**, and a documented reason for existing:

`correctness` · `faithfulness` · `completeness` · `instruction_following` · `tone` · `safety`

Two design choices worth calling out:

- **`tone` is isolated** so style cannot leak into `correctness`.
- **`safety` is scored bidirectionally** — over-refusal is penalised, not just harm.

**`passed` is computed in code**, never self-declared. The schema's `extra="forbid"` makes a
judge-emitted `passed` field *unparseable*, and the prompt forbids it explicitly. A self-declared
boolean is an uncalibrated threshold hiding inside a verdict.

### Malformed output

The repair loop is a **genuine repair**, not a retry: it appends the malformed text **and** the
parser's error, so the second request differs materially from the first. A `tenacity` decorator
re-issues identical arguments and structurally cannot do this — and the test asserts the request
changed, so a decorator-based implementation would fail it.

Truncation (`finish_reason == "length"`) is handled separately: the token budget **doubles**
(capped at 8192) and the model is asked for shorter rationales. Echoing the truncated text back
makes the prompt longer while the output budget stays fixed, guaranteeing every remaining attempt
also truncates — and since the cases that truncate are the verbose ones, that silently biases the
pass rate toward short answers.

---

## Statistics

Every estimator in [`src/stats.py`](src/stats.py) was verified against a reference implementation:

| Method | Verification |
|---|---|
| Quadratic-weighted Cohen's κ | Matches a from-scratch implementation to **1e-12** |
| Percentile bootstrap | Resampling unit is the **case**; seeded; degenerate resamples dropped |
| Wilson interval | Brute-forced over n≤100 × all x × 3 confidence levels — **0 mismatches** |
| Clopper-Pearson | Verified by its *defining* tail property — **0 violations** |
| ICC(1,1) | Matches `(MSB−MSW)/(MSB+(k−1)MSW)` exactly |
| Wilcoxon signed-rank | Ordinal-appropriate; zeros dropped; undefined case named |
| Exact McNemar | Two-sided binomial on discordant pairs only |
| Holm–Bonferroni | Genuinely step-down; adjusted p-values non-decreasing |

`labels` is always passed explicitly to κ. sklearn weights on a label's **position** in the array,
so inferring the class set from observed values collapses the gaps: with observed `{1, 2, 5}`, the
distance from 2 to 5 becomes 1 instead of 3, and a two-point disagreement is charged as one.

**Why Holm matters here:** ~15 comparisons at α=0.05 carry a family-wise error rate near 54%. At
these sample sizes the honest outcome is usually that most effects stop being significant — that
is a finding, not a weakness, and it is reported rather than quietly dropped.

---

## Audit trail

Every judge call is logged **per attempt**, including failures: prompt, raw response, token
counts, latency, cost. Secrets are redacted; the log is append-only.

```bash
python main.py replay <run_id> --stage run --report results/suite_report.json
```

`replay` rebuilds the report from `logs/verdicts.jsonl` **alone** and diffs it against the
committed file, exiting non-zero on mismatch. A committed result that does not follow from the log
is therefore a test failure rather than a matter of trust.

The effective mitigation set is recorded *in* each verdict, so replaying a `--mitigations off`
ablation run against an on-config still matches — previously it reported MISMATCH on a perfectly
intact log.

---

## Deployment gating

```bash
python main.py gate --baseline results/previous_validation.json
```

Gates on instrument **regression**, not absolute quality:

| Check | Blocking | Rule |
|---|---|---|
| Coverage | ✅ | Completed pairs ≥ 50% of the suite |
| Flip rate | ✅ | ≤ 0.20 — above it, the judge cannot resolve this suite |
| Probe regression | ✅ | No category below its baseline's **Wilson lower bound** |
| Gold agreement (κ) | ❌ advisory | Reported, never blocking |

Comparing probe **point estimates** would fire on ordinary sampling noise at 5–6 probes per
category; comparing against the interval only fires when the drop exceeds what noise explains.

Gold agreement is deliberately non-blocking. At n=15 with no established intra-rater ceiling, an
absolute quality bar is a number this design cannot defend — and a gate nobody trusts gets
switched off, after which nothing is gated at all.

---

## Testing

```bash
python -m pytest                        # 200 passed, 3 skipped — no API key, no network
ruff check . && ruff format --check .
```

| Suite | Tests | Covers |
|---|---:|---|
| `test_judge.py` | 22 | JSON recovery, genuine repair, truncation escalation, two error layers |
| `test_aggregator.py` | 20 | Mode selection, tallies, decision rule, coverage gate |
| `test_datasets.py` | 20 | Suite/gold/probe loading, unsafe-YAML refusal |
| `test_e2e_mocked.py` | 20 | Every CLI command through the real Typer layer with a fake provider |
| `test_schema.py` | 18 | Field order, score bounds, `extra="forbid"` |
| `test_suite_config.py` | 17 | Family derivation, cross-family invariant, routing-prefix trap |
| `test_stats.py` | 17 | κ `labels` behaviour, both halves of the claim |
| `test_mitigations.py` | 11 | Order swap, remap, exactly two calls per pair |
| `test_rate_limit.py` | 9 | TPM pacing, driven by a **fake clock** |
| `test_gating.py` | 8 | Gate blocking and non-blocking paths |

The 3 skips are `live`-marked and deliberately excluded from CI, so the suite never depends on a
provider being reachable.

---

## Cost and quota

Free tier both sides, so the real budget is **quota, not dollars**. Two limits bind, and neither
is the one people plan for:

- **Requests per day.** `gemini-3.5-flash` free tier allows **20/day** — measured from this
  pipeline's own audit log, since Google no longer publishes the figure. Against a validation
  suite needing ~370 calls that is ~19 days per run, which is why the judge runs on Groq.
- **Tokens per minute.** `llama-3.3-70b-versatile` allows **12,000 TPM** while one judge call
  costs ~4,600 tokens — about **2.6 calls/minute**, regardless of the 30 requests/minute the same
  tier advertises. Concurrency makes it worse, not better: two in-flight calls request ~9,200
  tokens at once and trip the limit. Measured — 7 of 18 cases lost at concurrency 2, 18 of 18 at
  concurrency 1.

Client-side token pacing reduced the refusal rate from **83% to 12%**.

---

## Limitations

Stated plainly, because a reviewer will find them anyway:

- **n=15 gold set.** A quadratic-weighted κ at that size routinely carries a 95% CI spanning most
  of the Landis–Koch range, from *fair* to *almost perfect*.
- **No washout re-label**, so the **ceiling** on judge–human agreement is unknown. κ 0.55 against a
  labeller whose own self-agreement is 0.60 is a completely different result from 0.55 against a
  labeller at 0.95, and this design cannot tell them apart. The code path is live — populate
  `overall_score_relabel` and it computes.
- **The sole labeller also wrote the outputs being labelled**, which probably makes these labels
  cleaner than a realistic human gold set, biasing agreement in a direction I cannot sign.
- **Self-enhancement confounds family with capability.** Two judges differ in more than lineage, so
  the win-rate delta is an upper bound, not an estimate. Disentangling it needs three judges
  spanning two families.
- **The A/B result above has insufficient coverage** — see [Results](#results).
- **The pooled criterion κ bootstraps at the wrong unit**: (case, criterion) rows are resampled
  independently, breaking within-case clustering, so that interval is anti-conservative. It is
  labelled secondary and descriptive for exactly this reason.

**Would I let this gate a release?** Not on absolute quality at this sample size — and I would
rather say that than perform confidence. As a **pre-merge signal** plus the two regression gates
above, yes.

---

<div align="center">
<sub>Built by <a href="https://github.com/LEKKALAGANESH">LEKKALA GANESH</a></sub>
</div>
