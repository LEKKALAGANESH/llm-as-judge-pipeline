# logs/

Generated at runtime and gitignored (`logs/*.jsonl`). Two append-only JSONL files
live here.

**`judge_calls.jsonl` -- one record per attempt.** Not per call, per *attempt*:
a case that needed a JSON repair leaves two records, and a case whose call failed
outright still leaves one. The record is written before parsing is attempted, so
a parse failure is auditable rather than invisible. Fields: `run_id`, `stage`,
`case_id`, `attempt`, `model`, `temperature`, the full `messages` list actually
sent, `raw_response`, `parse_ok`, `error`, `prompt_tokens`, `completion_tokens`,
`latency_ms`, `cached`, plus mode-specific extras (`order` for pairwise,
`reference_based` for pointwise, the delimiter `nonce`).

**`verdicts.jsonl` -- one record per case per run.** This is what `replay`
reconstructs a `SuiteReport` from and what `--resume` reads to skip work already
on disk.

Append-only is a property of the code, not a convention: nothing in
`src/audit_log.py` opens either file with anything but `"a"`, and no code path
rewrites or truncates one. A resumed run that re-judges a case appends a second
verdict record rather than editing the first; the reader takes the last one per
case. That is what makes the trail credible as an audit trail.

Any value of `GEMINI_API_KEY` / `GROQ_API_KEY` found in a prompt is replaced with
`<ENV_NAME:redacted>` before the record is written.

To commit a representative excerpt as evidence without committing the whole run:

```bash
head -n 3 logs/judge_calls.jsonl > logs/sample_judge_calls.jsonl
git add -f logs/sample_judge_calls.jsonl
```
