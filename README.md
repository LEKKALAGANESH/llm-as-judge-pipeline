# LLM-as-Judge Evaluation Pipeline

A pipeline that scores and compares model outputs with an LLM judge, and takes
the judge's own biases as seriously as the outputs it grades. Five named biases,
each with a code path that mitigates it and a number that measures it; three
independent judge-validation statistics; an A/B comparison decided by a rule
written before the run rather than after the results.

Judge: `groq/llama-3.3-70b-versatile` (Meta lineage). Candidates:
`groq/openai/gpt-oss-120b` and `groq/openai/gpt-oss-20b` (OpenAI lineage). That
split is architecture, not decoration -- it is the self-enhancement-bias
mitigation, and the pipeline refuses to start if it is violated. Note that both
now run on the *same provider*: the invariant is about model **lineage**, not
about who serves it, and `derive_family` reads the model portion rather than the
routing prefix precisely so that this distinction cannot be fudged.

The judge was `gemini/gemini-3.5-flash` until a measured constraint ruled it out.
Google no longer publishes free-tier limits, and this pipeline's own audit log
recorded `GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20` — **20
requests per day**, against a validation suite that needs roughly 370. That is
about 19 days per full run, so the free-tier posture this project claims was not
actually achievable on that judge. Groq meters the replacement at 1,000
requests/day. The binding limit there is **tokens** per minute, not requests: see
"Cost and quota" below.

## Results at a glance

**These cells are empty because I have not run the pipeline against the live
APIs, and I would rather ship a blank table than invented numbers.** Everything
needed to fill it is committed: the suites, the gold labels, the probes, the
commands, and the code that computes each figure. `results/README.md` lists the
four commands, in order, with the call budget.

| Figure | Value | Produced by | Lands in |
|---|---|---|---|
| Position-bias flip rate (raw) | pending run | `compare-configs` | `ab_comparison_report.json` -> `tally.flip_rate` |
| Position bias, noise-corrected | pending run | `compare-configs` | `position_bias_noise_floor.json` |
| Identical-pair non-Tie rate (noise floor) | pending run | `validate-judge` | `judge_validation_report.json` -> `probes.noise_floor_non_tie_rate` |
| Cohen's kappa_w vs gold, with 95% CI | pending run | `validate-judge` | `gold.overall_kappa` |
| Test-retest ICC(1,1) at T=0.7, k=5 | pending run | `validate-judge` | `test_retest.icc` |
| Adversarial pass rate per category (5 categories, Wilson CIs) | pending run | `validate-judge` | `probes.by_category` |
| Length-vs-score Spearman rho | pending run | `run-suite` | `suite_report.json` -> `length_vs_score_spearman` |
| Self-enhancement win-rate delta | pending run | `measure-self-enhancement` | `self_enhancement_report.json` |
| A/B winner, with counts | pending run | `compare-configs` | `winner` + `decision_trace` |
| Total judge calls / tokens / paid-tier equivalent | pending run | every command | `total_*` fields |

What I can state without a run, because these are properties of the code and are
tested: 178 tests pass with no API key and no network (3 more sit behind a `live`
marker and are deselected in CI); the report `replay` rebuilds from the log
matches the committed report field-for-field or the command exits non-zero; the
cross-family invariant refuses to run on a same-family pairing; and a pairwise
case costs exactly two judge calls, never more.

## Where each rubric line is satisfied

| Rubric line | Pts | Where |
|---|---|---|
| Pipeline correctness | 20 | `src/schema.py` (typed suite + verdicts), `src/judge.py` (repair loop, every attempt logged before return), `src/audit_log.py`, `main.py replay`. Tests: `tests/test_judge.py`, `tests/test_e2e_mocked.py` |
| Judging design | 20 | `config/rubric.yaml` (6 criteria, rationale + 1/3/5 anchors each), `src/aggregator.py::select_mode` (the one mode rule), [Judging modes](#judging-modes) below |
| Bias handling | 25 | [Bias handling](#bias-handling) below; `src/mitigations.py`, `src/ablation.py`, `data/test_suites/adversarial_probes.json`; `results/bias_ablation_report.json` |
| Judge validation | 20 | `src/validator.py`, `src/stats.py`, `data/gold_labels/human_annotated.json`; `results/judge_validation_report.json` |
| Comparison & engineering | 15 | `src/aggregator.py::decide_winner`, `config/suite_config.yaml`, cost/latency accounting in every report, `.github/workflows/ci.yml` |

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env                       # add GEMINI_API_KEY and GROQ_API_KEY
python main.py smoke-test                  # 2 calls: are both model ids still alive?
python main.py run-suite data/test_suites/general_qa.json
```

Then `compare-configs`, `measure-self-enhancement --allow-same-family`,
`validate-judge`, `run-ablation`. `python main.py --help` lists everything.
Tests: `pytest` (no keys needed) or `pytest -m live` (hits the real APIs).

## Judging modes

The brief names four modes. This pipeline implements them as two code paths, not
four, because reference-based and reference-free differ only in whether the
reference is rendered into the prompt. The selection rule lives in exactly one
place (`src/aggregator.py::select_mode`) and is unit-tested over all four
combinations:

```python
if comparing_two_configs:                    return "pairwise"
elif test_case.expected_output is not None:  return "pointwise_reference_based"
else:                                        return "pointwise_reference_free"
```

**Pointwise** is what a release gate needs: an absolute bar on a fixed scale,
vulnerable to score clustering (hence the anchors, and the clustering measurement
in the ablation). **Pairwise** is what a regression test needs, since "did this
get better or worse" has no absolute answer; it sidesteps calibration but is
exposed to position bias, hence the order swap. **Reference-based** fits factual
QA where a ground truth exists, and the prompt explicitly tells the judge that a
different valid phrasing is not an error. **Reference-free** fits open-ended
generation and leans hardest on the judge's own calibration.

## The rubric

Six criteria in `config/rubric.yaml`, each with a description, a stated rationale
for existing, and worked anchors at 1, 3 and 5. Editing the rubric never means
opening a `.py` file.

| Criterion | Why it is separate |
|---|---|
| correctness | The one a gate cannot do without. Split from faithfulness because an answer can be true but unsupported, or supported but false, and those need different fixes. |
| faithfulness | Catches hallucination specifically, as opposed to being merely wrong. Most sensitive to padding an answer with plausible unsourced detail -- the failure verbosity bias rewards. |
| completeness | Deliberately narrow: coverage of what was asked, never volume. Paired with the anti-verbosity clause so it cannot become a back door for length. |
| instruction_following | The criterion most often silently traded away for helpfulness, and the one surrounding software actually depends on. A violated explicit constraint caps it at 2. |
| tone | Isolated into its own low-stakes box precisely so style cannot leak into correctness. The structural half of the sycophancy mitigation. |
| safety | Scored in both directions. A rubric that only punishes unsafe output rewards a model that refuses everything, which is the cheapest way to game it. |

`passed` is computed in code -- `overall_score >= pass_threshold AND
min(criterion scores) >= min_criterion_score`, both from config -- and is **not**
in the judge's output schema. A judge-declared boolean would be an uncalibrated
fourth judgment with no anchors, and it would make the bar un-retunable without
re-spending the whole suite. Because the aggregator recomputes it, `replay` can
answer "what would the pass rate have been at threshold 3?" from the log alone.

## Bias handling

Three checkboxes per bias: named, mitigated in code, measured. A README paragraph
without a code path and a number satisfies none of them.

| Bias | Mitigated in code by | Measured by |
|---|---|---|
| Position | `mitigations.evaluate_pairwise_unbiased` runs every pair in both orders and resolves disagreement to `Tie (Position Inconsistency)` | flip rate = inconsistent / completed pairs, **and** the noise-corrected estimate `flip_rate(swapped) - flip_rate(same-order repeat)`, **and** the non-Tie rate over byte-identical pairs |
| Verbosity | anti-verbosity clause in `rubric.yaml` (ablatable), padded-vs-concise probes | Spearman rho between `len(model_output)` and `overall_score` across the suite, with and without the clause (zero extra API calls -- the lengths and scores are already logged), plus the padded-probe pass rate in both conditions |
| Self-enhancement | judge family != every candidate family, enforced as a hard startup assertion on the **model** portion of the id | win-rate delta between the cross-family judge and `groq/openai/gpt-oss-120b` judging identical content |
| Sycophancy / style | `rationale` precedes `score` in the schema, so under constrained decoding the evidence must be emitted first; confidently-wrong and terse-but-correct probes | per-category probe pass rate with Wilson intervals (6 sycophancy probes, 5 prompt-injection probes) |
| Score clustering | few-shot anchors at 1/3/5 per criterion (ablatable) | overall-score std dev, modal share, Shannon entropy and 4-or-5 share, with anchors and without |

Two of the five are not flag-ablatable and the report says so rather than
implying an ablation that did not happen. Rationale-first is the schema's field
order; turning it off means shipping a second schema, so the probes measure it
instead. Cross-family judging is which models you call; its "before" condition is
a different judge, which is why it has its own command.

### Why the field order is the mitigation

Under constrained decoding a model emits JSON fields in schema order. A schema
declaring `score` before `rationale` makes the judge commit to a number before
generating any reasoning, and the rationale becomes post-hoc rationalisation --
which silently voids the grounding mitigation while every prompt still *claims*
it. So `CriterionScore` is `(rationale, score)`, `PointwiseVerdict` is
`(criteria_breakdown, overall_rationale, overall_score)`, `PairwiseVerdict` is
`(rationale, winner)`, the wire schema sent to the provider is generated from
those models rather than hand-written, and there is a test asserting
`list(CriterionScore.model_fields) == ["rationale", "score"]`.

### Why family, not provider

Under litellm the routing prefix is not the model lineage. `groq/openai/gpt-oss-120b`
is routed by Groq and is family **openai**; `bedrock/anthropic.claude-3` is routed
by Bedrock and is family **anthropic**. A check written against the routing prefix
would compare "gemini" to "groq", pass for the wrong reason, and keep passing
after someone swapped the judge to a Groq-served Gemini -- exactly the pairing the
invariant exists to prevent. `derive_family` matches on the model portion, and
`all()` over an empty candidate list is guarded too, since it is vacuously true:
a non-empty `candidates` list or an explicit `generator_family:` is required
before the mitigation may be claimed. The resolved families are written into every
report so the claim is auditable rather than asserted. The deliberate same-family
run needs `--allow-same-family`, which is recorded as a waiver in the report.

### Position bias: a worked example

Case `cmp-03`, "Is it safe to run `git push --force` on a shared branch?"

* **Forward call** -- A = the hedging v1 answer, B = the direct v2 answer. Judge
  returns `{"rationale": "...B states the consequence and names --force-with-lease...", "winner": "Model_B"}`.
* **Reverse call** -- the same case with the outputs swapped, so the v2 answer is
  now presented as `Model_A`. Judge returns `{"rationale": "...", "winner": "Model_A"}`.
* **Remap** -- a reverse verdict of `Model_A` means the output shown second in the
  original frame won, i.e. `Model_B`. `remap_reverse_winner("Model_A") -> "Model_B"`.
* **Resolve** -- forward `Model_B` == remapped `Model_B`, so `final_winner = "Model_B"`,
  `position_consistent = true`, and the pair does not count toward the flip rate.

Had the reverse call returned `Model_B` instead, the remap would give `Model_A`,
the two orders would disagree, and the pair would resolve to
`Tie (Position Inconsistency)`: excluded from both win counts, counted in the
flip rate, and visible in the report. If either order fails after retries the
pair is excluded from the win counts **and** from the flip-rate denominator and
counted as `incomplete_pairs` -- never resolved from the one surviving order,
which would silently reintroduce the bias the protocol exists to remove.

## Judge validation

Three independent numbers, because they answer different questions and a judge
can be consistently wrong or accurate-on-average but noisy case by case.

**Agreement.** 15 hand-labelled cases in `data/gold_labels/human_annotated.json`.
I labelled them myself, against the same rubric, one criterion at a time, before
running any judge -- labels chosen after seeing judge output turn validation into
confirmation. The file discloses the labeller, the procedure and four caveats,
including the one that matters most: I also authored the outputs being labelled,
so the intended defect is more salient to me than it would be to a blind rater.

Kappa is quadratic-weighted with `labels=[1,2,3,4,5]` always passed explicitly,
and never reported alone: alongside it go the 5x5 confusion matrix, both marginal
distributions, raw percent agreement, within-one agreement, Spearman rho, and a
2,000-resample percentile bootstrap CI. The interpretation rule is pre-committed
in `stats.py::interpret_kappa` rather than chosen after seeing the number -- low
kappa + high raw agreement + concentrated marginals is marginal skew, not judge
failure; low kappa + high rho means mis-calibrated (fixable by moving the anchors
or the threshold); low kappa + low rho means unreliable (not fixable by
calibration). A rater with fewer than two distinct labels yields `kappa: null`
and a stated reason, never `NaN`. At n=15 the interval dominates the estimate --
a +-0.10 half-width needs roughly 100 items -- so the interval is the result.

**Test-retest.** The same suite k=5 times at T=0.7, reporting ICC(1,1) on the
unrounded overall score first and the pass/fail flip rate second, labelled as the
threshold artifact it is: a judge with tiny noise sitting on the pass boundary
shows a huge flip rate, and one with large noise far from it shows zero. The
cache is bypassed for these calls, since a cache keyed on the prompt would return
byte-identical responses and manufacture a perfect 0% flip rate -- the meaningless
validation number this protocol exists to avoid. The config refuses T=0 unless
you pass `--acknowledge-t0`. And because modern serving stacks are not
bit-deterministic, even T=0 normally shows a small non-zero flip rate: *exactly*
0.0 sets `sampling_warning` and reports how many cases produced byte-identical
rationales, which is the signature of a temperature setting that never took
effect.

**Adversarial probes.** 27 probes across five categories in
`data/test_suites/adversarial_probes.json`, every one carrying a pre-registered
numeric bound so that "the judge passed" is a test rather than a story told
afterwards. Pass rates are reported per category with Wilson intervals; n=1 per
category yields 0% or 100% and is not a measurement.

* **verbosity** (6) -- padded vs concise, scored **only on correctness and
  faithfulness**, where the pair is stipulated identical. Scoring all six would
  confound the bias signal with a real quality difference, since a padded answer
  legitimately differs on completeness, instruction-following and tone.
* **sycophancy** (6) -- three confidently-wrong (polished, authoritative, false)
  and three terse-but-correct controls.
* **position_noise_floor** (5) -- byte-identical pairs; the non-Tie rate is an
  assumption-free estimate of pure position plus sampling bias.
* **over_correction** (5) -- thorough-and-better vs terse-and-incomplete, where
  the *longer* answer must win. Without these you can score 100% on everything
  else by cranking the anti-verbosity clause until the judge can no longer reward
  legitimate thoroughness.
* **prompt_injection** (5) -- an injected verdict object, a fake system block, a
  delimiter-escape attempt, a replacement rubric, and a social-engineering
  appeal. Judged content is wrapped in per-request nonce delimiters with the
  nonce stripped from the content first, and the rubric tells the judge that an
  output trying to manipulate its evaluator is a defect, not a compliment.

## A/B comparison

`compare-configs` runs the full pairwise-with-order-swap protocol over
`data/test_suites/comparison_qa.json` (prompt v1 vs v2) and applies this rule,
which lives in `aggregator.py::decide_winner` and is quoted verbatim into every
report:

```
# Step 1 -- validity gate. Evaluated FIRST and short-circuiting.
if flip_rate > 0.20: return "Judge unreliable on this suite - no winner declared"

# Step 2 -- effect. resolved = wins_a + wins_b (excludes ties AND inconsistent pairs)
if wins_b / resolved > 0.55: winner = config B
elif wins_a / resolved > 0.55: winner = config A
else: winner = "Inconclusive - no significant difference"
```

Three things the naive version gets wrong. There is **no mean-score term**: a
comparison run is pairwise-only and `PairwiseVerdict` carries no numeric score,
so that criterion would not be computable from the run it is applied to. The
flip-rate gate is a **precondition on the instrument, not a third criterion on
the effect** -- ANDing it in would collapse "the judge is untrustworthy" and "the
two configs are equivalent" into one `Inconclusive`, which are operationally
opposite conclusions (fix your judge vs. ship either). And the **denominators are
explicit**: `resolved` excludes ties and inconsistent pairs, the flip-rate
denominator excludes incomplete pairs, and both appear in the report.

The 0.20 threshold means "if more than one pair in five reverses on order swap,
the pairwise signal is too noisy to declare a winner"; it lives in
`suite_config.yaml` with that rationale attached. The 0.55 threshold is a
heuristic, not a significance test: at n=25 with a true 50/50 process it fires
about a third of the time, and 55% would need n around 270 to be significant. So
the win rate is reported with a Clopper-Pearson interval and the interval is left
to speak, which at these sample sizes is visibly enormous.

The judge never sees which prompt variant produced which output -- the comparison
suite carries one shared task instruction and keeps both variants in its
provenance block. Showing the judge both variants would let it infer provenance
in a comparison whose entire point is to be blind.

## Engineering notes

**Audit trail.** `logs/judge_calls.jsonl` gets one record per *attempt* --
repairs and failures included -- with the full message list, the raw response,
token counts and `latency_ms`, written before parsing is attempted. Both logs are
append-only by construction. `main.py replay <run_id>` rebuilds the `SuiteReport`
from `logs/verdicts.jsonl` alone and diffs it against the committed report,
ignoring only `generated_at` and wall-clock latency; a mismatch exits 2. That
turns "auditable and replayable" into a property with a test, and the same
reconstruction backs `--resume` after a mid-run 429.

**Two error layers, deliberately not merged.** A `tenacity` decorator cannot
repair JSON: it re-invokes with identical arguments, so the "repair" is a
re-issue of the prompt that just failed, and a test asserting only "it eventually
succeeded" passes anyway. The repair loop is an ordinary bounded loop that
appends the malformed output **and** the parser's error to the message list; the
test asserts the second request body contains both. Transient failures (429, 5xx,
connection) are retried by `tenacity` one layer down. Auth, permission and
bad-request failures are never retried -- inside a per-case loop those retries
multiply one misconfiguration across the whole suite -- and the error names which
provider's key to check.

**Cost.** Free tier both sides, so the real budget is quota, not dollars. A disk
cache keyed on `sha256(model, messages, params)` makes development re-runs free;
`--max-calls-per-run` and `--max-cost-usd` are circuit breakers checked *before*
each request; **reasoning** models set `reasoning_effort="low"` and
`include_reasoning=False`, since reasoning tokens bill as output and at default
effort can halve a day's budget. That is gated on the model, not the provider --
Groq also serves non-reasoning models that reject those parameters with a fatal
400. Tokens are summed across every attempt, so repaired calls are not invisible,
and each report prints what the run *would* have cost on a paid tier. "We did not
track cost because it was free" misses the point of the requirement.

**Quota, measured rather than assumed.** Two limits bind, and neither is the one
people plan for:

- **Requests per day.** `gemini-3.5-flash` free tier allows **20/day**, which is
  why it is not the judge. Groq allows 1,000/day for the model that is.
- **Tokens per minute.** This is the real ceiling for this workload.
  `llama-3.3-70b-versatile` allows **12,000 TPM** while one judge call costs
  ~4,600 tokens — about **2.6 calls/minute**, regardless of the 30 requests/minute
  the same tier advertises. Running two calls concurrently simply requests ~9,200
  tokens at once and trips the limit: a measured run lost 7 of 18 cases that way,
  and the same suite scored 18 of 18 at concurrency 1. `max_concurrent_judge_calls`
  is therefore 1, and the retry backoff ceiling exceeds the longest `retryDelay`
  either provider asks for.

**Config hygiene and data policy.** Model names in YAML, keys only in `.env`
(gitignored), `yaml.safe_load` everywhere. Free tiers generally reserve the right
to train on submitted content and to permit human review, so nothing confidential
should go through this pipeline; these suites are synthetic and mine. Google's
free tier additionally is unavailable in the EEA, Switzerland and the UK — less
binding now that the judge runs on Groq, but still relevant to anyone reproducing
the Gemini comparison in `measure-self-enhancement`.

## Discussion

**How biased is the judge before vs after?** `run-ablation` is the only thing
that answers this, and it answers it per bias: it runs the suite with the three
ablatable mitigations on and off and emits `bias_ablation_report.json` with a
before/after column each. For position bias, the "before" condition is a
single-order run, which always produces a decisive-looking winner -- the flip rate
is exactly the share of those winners that were artifacts. For verbosity, it is
the length-score Spearman rho with and without the clause. For clustering, it is
the score distribution with and without anchors. I have not run it, so I am not
going to characterise the direction or size of any of those deltas here.

**Would I let this gate a release?** Not on the numbers this design can produce
at this sample size, and I would rather say that than perform confidence. Three
reasons, all structural rather than contingent on how the run turns out.

First, n=15 for the gold set is not a decision-grade sample. A quadratic-weighted
kappa at that size routinely carries a 95% CI spanning most of the Landis-Koch
range, from *fair* to *almost perfect*. A gate needs to know which one it has.

Second, my gold labels have no established ceiling. I did not perform the washout
re-label, so intra-rater reliability is unknown, and kappa 0.55 against a labeller
whose own self-agreement is 0.60 is a completely different result from 0.55
against a labeller at 0.95. The design cannot distinguish them. The code path is
live -- populate `overall_score_relabel` and it computes -- so this is a gap I
chose not to close in the time available, not one I could not close.

Third, the sole labeller also wrote the outputs being labelled. That makes the
intended defect in each case more salient to me than it would be to a blind
rater, which probably makes these labels cleaner and more internally consistent
than a realistic human gold set. It biases the agreement measurement in a
direction I cannot sign.

What I would do instead: use it as a **pre-merge signal, not a gate** -- surface
the verdict on the PR, block on nothing. Then gate on the two things that are
cheap and unambiguous even with an imperfect judge: a hard stop if the flip rate
exceeds 0.20 (the judge is telling you it cannot resolve this suite), and a hard
stop on any adversarial probe category whose pass rate drops below its previous
run's lower confidence bound. Both are regression signals about the *instrument*,
which is a much easier thing to be confident about than an absolute quality bar.

**A weakness I would fix first.** The self-enhancement measurement confounds
family with capability. `llama-3.3-70b-versatile` and `gpt-oss-120b` differ in more
than lineage, so a win-rate delta between them is an upper bound on self-enhancement
rather than an estimate of it -- some of it is simply two different judges having
different opinions. Disentangling that needs at least three judges spanning two
families each, which is the ensemble improvement I did not build. I say so in the
report body rather than in a footnote, because a number presented without its
confound is worse than no number.

**Second weakness.** `comparison_qa.json` currently ships hand-authored stand-in
outputs, not real `gpt-oss-120b` generations, because I have not run the generator
with live credentials. The file says so in its provenance block, every case is
tagged `source: hand_authored`, and `scripts/generate_candidates.py` regenerates
it with real output and stamps the model id and timestamp. Until that is run, any
A/B result is a demonstration of the comparison machinery, not a finding about
gpt-oss-120b.

## Layout and reproducibility

```
config/     rubric.yaml (6 criteria, anchors, clauses) + suite_config.yaml
data/       test_suites/{general_qa, comparison_qa, adversarial_probes}.json
            gold_labels/human_annotated.json
src/        schema  errors  suite_config  audit_log  llm_client  judge
            mitigations  runner  aggregator  validator  stats  ablation  reports
scripts/    generate_candidates.py  (run once, commit the output)
tests/      178 offline tests + 3 behind the `live` marker
main.py     Typer CLI
```

Dependency direction is enforced, not just documented: `schema` and `errors` have
no internal imports; `judge` never imports `aggregator` or `validator`;
`aggregator` is a pure `list[Verdict] -> SuiteReport` reduction with zero I/O and
zero LLM calls, which is precisely why the pass-rate, win-rate and flip-rate
arithmetic behind the two largest rubric lines is unit-tested against
hand-computed fixtures.

Every report records the run id, model id, temperature, resolved model families,
which mitigations were active, and the exact decision-rule text. Record the run
date beside any committed number: LLM-judge figures move between runs, and a
reader who re-runs and sees different ones with no warning will reasonably
conclude the first set was invented.
