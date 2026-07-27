"""Generate the A/B candidate outputs ONCE and commit them.

This is a setup step, not part of the evaluation loop, and the distinction is a
quota decision rather than a stylistic one. Groq's free tier allows roughly 111
calls per day at this workload's token size (200K TPD / ~1,800 tokens). Producing
15 cases x 2 prompt variants costs 30 calls once and fits comfortably;
regenerating on every evaluation run does not, and would also make every A/B
result incomparable with the last one.

    python scripts/generate_candidates.py --out data/test_suites/comparison_qa.json

Both prompt variants below are recorded into the output file's provenance block,
so the committed suite always states exactly which prompts produced it. The judge
is never shown either variant -- only the shared task instruction -- because
showing it both would let it infer which arm produced which output.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.datasets import load_comparison_suite  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.suite_config import check_api_key, load_config  # noqa: E402

PROMPT_VARIANTS: dict[str, str] = {
    "v1": "You are a helpful assistant. Answer the user's question.",
    "v2": (
        "You are a helpful assistant. Answer the user's question directly in the first "
        "sentence, then give at most two sentences of supporting detail. Say explicitly when "
        "you are uncertain or when the answer depends on something you were not told. Do not "
        "write a preamble, restate the question, or add a closing summary."
    ),
}

SHARED_TASK_INSTRUCTION = "Answer the user's question."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/suite_config.yaml")
    parser.add_argument("--out", default="data/test_suites/comparison_qa.json")
    parser.add_argument(
        "--seed",
        default="data/test_suites/comparison_qa.json",
        help="File whose `input` fields are reused as the questions, so regeneration keeps the "
        "same suite and only the outputs change.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan and the call count, call nothing."
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config, enforce=False)
    seed = load_comparison_suite(args.seed)
    cases = seed.cases[: args.limit] if args.limit else seed.cases

    if len(config.candidates) < 2:
        print("suite_config.yaml needs two entries under `candidates:` (config_a and config_b).")
        return 1
    cand_a, cand_b = config.candidates[0], config.candidates[1]
    for cand in (cand_a, cand_b):
        if cand.prompt_variant not in PROMPT_VARIANTS:
            print(f"unknown prompt_variant {cand.prompt_variant!r} for candidate {cand.label!r}")
            return 1

    print(f"{len(cases)} question(s) x 2 configs = {2 * len(cases)} generation calls")
    print(f"  A = {cand_a.label}: {cand_a.model} with prompt {cand_a.prompt_variant}")
    print(f"  B = {cand_b.label}: {cand_b.model} with prompt {cand_b.prompt_variant}")
    if args.dry_run:
        print("dry run: nothing called, nothing written.")
        return 0

    check_api_key(cand_a.model)
    check_api_key(cand_b.model)
    client = LLMClient(
        cache_path=config.run.cache_path,
        max_calls=4 * len(cases),
        price_lookup=lambda m: (config.price_for(m).input, config.price_for(m).output),
    )

    def generate(model: str, variant: str, question: str) -> str:
        resp = client.complete(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT_VARIANTS[variant]},
                {"role": "user", "content": question},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return resp.text.strip()

    stamp = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    out_cases = []
    for i, case in enumerate(cases, start=1):
        print(f"  [{i}/{len(cases)}] {case.case_id}: {case.input[:60]}...")
        out_cases.append(
            {
                "case_id": case.case_id,
                "input": case.input,
                "system_prompt": SHARED_TASK_INSTRUCTION,
                "output_a": generate(cand_a.model, cand_a.prompt_variant or "v1", case.input),
                "output_b": generate(cand_b.model, cand_b.prompt_variant or "v2", case.input),
                "config_a_label": cand_a.label,
                "config_b_label": cand_b.label,
                "expected_output": case.expected_output,
                "tags": [],
                "source": f"{cand_a.model} / {cand_b.model} @ {stamp}",
            }
        )

    payload = {
        "suite_id": seed.suite_id,
        "description": seed.description,
        "config_a_label": cand_a.label,
        "config_b_label": cand_b.label,
        "provenance": {
            "generator_model_a": cand_a.model,
            "generator_model_b": cand_b.model,
            "generator_family": config.generator_family,
            "config_a_prompt": PROMPT_VARIANTS[cand_a.prompt_variant or "v1"],
            "config_b_prompt": PROMPT_VARIANTS[cand_b.prompt_variant or "v2"],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "generated_at": stamp,
            "source": "REAL MODEL OUTPUT generated by scripts/generate_candidates.py",
            "regenerate_command": f"python scripts/generate_candidates.py --out {args.out}",
        },
        "cases": out_cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out} ({len(out_cases)} pairs)")
    print(
        f"generation used {client.api_calls} api call(s), "
        f"{client.prompt_tokens} in / {client.completion_tokens} out tokens"
    )
    print("Commit this file. Do not regenerate it as part of an evaluation run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
