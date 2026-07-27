# results/

Tracked, not gitignored: these files are the deliverable evidence of a run, and
`replay` checks a committed report against the log that produced it.

Empty right now, on purpose. Every file this directory is meant to hold is the
output of a real run against real provider APIs, and I have not fabricated any
of them.

Producing them requires `GEMINI_API_KEY` and `GROQ_API_KEY` in `.env` (see
`.env.example`). The four commands below fill the directory; each writes the JSON
and prints a human-readable summary to stdout.

| File | Command that produces it | What it carries |
|---|---|---|
| `suite_report.json` | `python main.py run-suite data/test_suites/general_qa.json` | pass rate, mean score per criterion, score distribution, length-vs-score Spearman, tokens, latency |
| `ab_comparison_report.json` | `python main.py compare-configs` | win/loss/tie/inconsistent counts, flip rate, win rate with a Clopper-Pearson CI, the declared winner and the decision trace |
| `position_bias_noise_floor.json` | `python main.py compare-configs` (with `--noise-floor`, the default) | same-order repeat flip rate, and the noise-corrected position-bias estimate |
| `judge_validation_report.json` | `python main.py validate-judge` | quadratic-weighted kappa with a bootstrap CI plus its companion statistics, test-retest ICC at T>0, per-category adversarial probe pass rates with Wilson intervals |
| `self_enhancement_report.json` | `python main.py measure-self-enhancement --allow-same-family` | cross-family vs same-family win rates and the delta |
| `bias_ablation_report.json` | `python main.py run-ablation` | the before/after row per bias -- the only source of the "how biased before vs after" numbers |

Recommended order: `compare-configs` and `measure-self-enhancement` before
`validate-judge`, because `validate-judge` folds the noise floor and the
self-enhancement delta into its report if those files already exist, and notes
their absence in `caveats` if they do not.

Budget roughly 370 judge calls for the full set, about 270 on Gemini and 100 on
Groq. Read your actual Gemini RPD at <https://aistudio.google.com/rate-limit>
first -- the design fits comfortably at 1,000+ RPD and fails outright at 250, and
Google no longer publishes the number. `python main.py smoke-test` checks both
providers in two calls before you spend anything.

Every run is resumable and every result is replayable:

```
python main.py run-suite <suite> --resume <run_id>          # continue after a 429
python main.py replay <run_id> --stage run --report results/suite_report.json
```

`replay` rebuilds the report from `logs/verdicts.jsonl` alone and diffs it
against the committed file, so a committed result that does not follow from the
log is a test failure rather than a matter of trust.

When you do commit results, record the model id, temperature and run date beside
them. LLM-judge numbers move between runs, and a reader who re-runs and sees
different figures with no warning will reasonably conclude the results were made
up.
