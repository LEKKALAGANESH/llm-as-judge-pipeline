"""Canned judge replies, good and bad. No API key, no network, no mocking of the
parser itself -- these strings are fed through the real parse path.

The four failure modes named in the design are all here: a missing field, a wrong
type, truncated JSON, and extra prose wrapping the JSON. The last two categories
are the interesting ones, because a well-written extractor should RECOVER from
prose-wrapping and fenced JSON without spending a repair round trip, while
truncation genuinely needs the repair loop.
"""

from __future__ import annotations

CRITERIA = [
    "correctness",
    "faithfulness",
    "completeness",
    "instruction_following",
    "tone",
    "safety",
]


def valid_pointwise(
    *,
    scores: dict[str, int] | None = None,
    overall: int = 4,
    rationale: str = "Cites the specific span 'signed on 28 June 1919', which matches the reference.",
) -> str:
    scores = scores or dict.fromkeys(CRITERIA, 4)
    body = ", ".join(
        f'"{name}": {{"rationale": "{rationale}", "score": {scores.get(name, 4)}}}'
        for name in CRITERIA
    )
    return (
        f'{{"criteria_breakdown": {{{body}}}, '
        f'"overall_rationale": "{rationale}", "overall_score": {overall}}}'
    )


def valid_pairwise(
    winner: str = "Model_B", rationale: str = "B is more direct on correctness."
) -> str:
    return f'{{"rationale": "{rationale}", "winner": "{winner}"}}'


# -- recoverable without a repair round trip -------------------------------

FENCED_JSON = "```json\n" + valid_pointwise() + "\n```"

PROSE_WRAPPED = (
    "Sure! Here is my evaluation of the output you provided:\n\n"
    + valid_pointwise()
    + "\n\nLet me know if you would like me to explain any of the scores."
)

# -- genuinely malformed: these must drive the repair loop -----------------

TRUNCATED = valid_pointwise()[: len(valid_pointwise()) // 2]

MISSING_OVERALL_SCORE = (
    '{"criteria_breakdown": {'
    + ", ".join(f'"{c}": {{"rationale": "ok", "score": 4}}' for c in CRITERIA)
    + '}, "overall_rationale": "Solid answer."}'
)

WRONG_TYPE = (
    '{"criteria_breakdown": {'
    + ", ".join(f'"{c}": {{"rationale": "ok", "score": "four"}}' for c in CRITERIA)
    + '}, "overall_rationale": "Solid answer.", "overall_score": 4}'
)

SCORE_OUT_OF_RANGE = (
    '{"criteria_breakdown": {'
    + ", ".join(f'"{c}": {{"rationale": "ok", "score": 7}}' for c in CRITERIA)
    + '}, "overall_rationale": "Solid answer.", "overall_score": 7}'
)

MISSING_CRITERION = (
    '{"criteria_breakdown": {'
    + ", ".join(f'"{c}": {{"rationale": "ok", "score": 4}}' for c in CRITERIA[:-1])
    + '}, "overall_rationale": "Solid answer.", "overall_score": 4}'
)

EXTRA_CRITERION = (
    '{"criteria_breakdown": {'
    + ", ".join(f'"{c}": {{"rationale": "ok", "score": 4}}' for c in CRITERIA)
    + ', "vibes": {"rationale": "good vibes", "score": 5}'
    + '}, "overall_rationale": "Solid answer.", "overall_score": 4}'
)

EMPTY_RATIONALE = (
    '{"criteria_breakdown": {'
    + ", ".join(f'"{c}": {{"rationale": "", "score": 4}}' for c in CRITERIA)
    + '}, "overall_rationale": "Solid answer.", "overall_score": 4}'
)

NO_JSON_AT_ALL = "I am not able to evaluate this output because it appears to be empty."

EMPTY_RESPONSE = ""

PAIRWISE_BAD_WINNER = '{"rationale": "A wins.", "winner": "Model_C"}'

REPAIRABLE = [
    ("truncated", TRUNCATED),
    ("missing_field", MISSING_OVERALL_SCORE),
    ("wrong_type", WRONG_TYPE),
    ("score_out_of_range", SCORE_OUT_OF_RANGE),
    ("missing_criterion", MISSING_CRITERION),
    ("extra_criterion", EXTRA_CRITERION),
    ("empty_rationale", EMPTY_RATIONALE),
    ("no_json", NO_JSON_AT_ALL),
    ("empty_response", EMPTY_RESPONSE),
]

RECOVERABLE = [
    ("fenced", FENCED_JSON),
    ("prose_wrapped", PROSE_WRAPPED),
    ("plain", valid_pointwise()),
]
