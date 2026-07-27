"""Bias mitigations that change how the judge is *called*, not what it is told.

The position-bias protocol lives entirely here and is invoked once from the
runner, so no layer above this one knows or cares that a single pairwise case
costs two judge calls.

The one-order-failed case is specified rather than left to the implementer,
because all three plausible guesses change the headline numbers: resolving from
the single successful order silently reintroduces the exact position bias the
protocol removes; counting it inconsistent inflates the flip rate; dropping it
shrinks the denominator with no record. The rule implemented here is: if either
order fails, the pair is excluded from win/loss/tie AND from the flip-rate
denominator, and is counted separately as an incomplete pair.
"""

from __future__ import annotations

from collections.abc import Callable

from .judge import JudgeOutcome
from .schema import CallAccounting, PairwiseCaseResult, PairwiseWinner, TestCase

# A judge-pair callable: (first_output, second_output, order_label) -> outcome.
JudgePairFn = Callable[[str, str, str], JudgeOutcome]

POSITION_INCONSISTENT = "Tie (Position Inconsistency)"


def remap_reverse_winner(winner: PairwiseWinner) -> PairwiseWinner:
    """Translate a reversed-order verdict back into the original (A, B) frame.

    In the reverse call the judge saw output_b as "Model_A", so a reverse
    verdict of Model_A means output_b won, which is Model_B in the original
    frame. Ties are frame-invariant.
    """
    if winner == "Model_A":
        return "Model_B"
    if winner == "Model_B":
        return "Model_A"
    return "Tie"


def evaluate_pairwise_unbiased(
    judge_pair_fn: JudgePairFn,
    *,
    case_id: str,
    output_a: str,
    output_b: str,
    order_swap: bool = True,
    judge_model: str = "",
    judge_temperature: float = 0.0,
    run_id: str = "",
    tags: list[str] | None = None,
) -> PairwiseCaseResult:
    """Judge one pair in both orders and resolve the two verdicts."""
    result = PairwiseCaseResult(
        case_id=case_id,
        order_swapped=order_swap,
        length_a=len(output_a),
        length_b=len(output_b),
        judge_model=judge_model,
        judge_temperature=judge_temperature,
        run_id=run_id,
        tags=list(tags or []),
    )

    forward = judge_pair_fn(output_a, output_b, "forward")
    accounting = forward.accounting
    result.forward_rationale = getattr(forward.verdict, "rationale", None)
    if forward.verdict is not None:
        result.forward_winner = forward.verdict.winner  # type: ignore[union-attr]

    if not order_swap:
        # Ablation branch: the "before" condition for position bias. A single
        # order always yields a decisive-looking winner -- that is the point of
        # the measurement, not an oversight.
        result.accounting = accounting
        if forward.verdict is None:
            result.incomplete = True
            result.failure = forward.failure
            result.error = forward.error
            return result
        result.final_winner = result.forward_winner
        result.position_consistent = None
        return result

    reverse = judge_pair_fn(output_b, output_a, "reverse")
    accounting = accounting.merged(reverse.accounting)
    result.accounting = accounting
    result.reverse_rationale = getattr(reverse.verdict, "rationale", None)

    if forward.verdict is None or reverse.verdict is None:
        result.incomplete = True
        failed = forward if forward.verdict is None else reverse
        result.failure = failed.failure
        which = "forward" if forward.verdict is None else "reverse"
        result.error = f"{which} order failed: {failed.error}"
        return result

    remapped = remap_reverse_winner(reverse.verdict.winner)  # type: ignore[union-attr]
    result.reverse_winner = remapped

    if result.forward_winner == remapped:
        result.final_winner = result.forward_winner
        result.position_consistent = True
    else:
        result.final_winner = POSITION_INCONSISTENT
        result.position_consistent = False
    return result


# --------------------------------------------------------------------------
# Probe construction
# --------------------------------------------------------------------------

PADDING_PREAMBLE = (
    "That is an excellent and very important question, and I am glad you asked it. "
    "Before answering, it is worth setting out that this is a topic with a good deal "
    "of nuance, and reasonable people approach it from several angles. With that "
    "context established, here is a thorough treatment of your question.\n\n"
)

PADDING_CODA = (
    "\n\nTo summarise the above: the key points are exactly the ones already stated, "
    "restated here for clarity and completeness. It is always worth remembering that "
    "context matters, that individual circumstances vary, and that you may wish to "
    "consult additional sources for further detail. I hope this comprehensive and "
    "detailed answer has been helpful to you."
)


def build_padded_probe(base_case: TestCase, *, suffix: str = "-padded") -> TestCase:
    """Same substantive content, more words.

    The delta between this and the concise baseline on **correctness and
    faithfulness only** is the verbosity signal. The other four criteria are
    deliberately excluded from the comparison because a padded answer legitimately
    differs on completeness, instruction-following and tone -- scoring the pair on
    all six would confound the measurement with real quality differences.
    """
    return base_case.model_copy(
        update={
            "case_id": f"{base_case.case_id}{suffix}",
            "model_output": PADDING_PREAMBLE + base_case.model_output + PADDING_CODA,
            "tags": [*base_case.tags, "probe:padded"],
        }
    )


CONFIDENT_FRAMING = "This is well established and there is no real debate about it. "

CONFIDENT_CODA = "\n\nThis is the standard account and is uncontroversial among specialists."


def build_confidently_wrong_probe(
    base_case: TestCase,
    *,
    false_claim: str,
    suffix: str = "-confidently-wrong",
) -> TestCase:
    """Polished, fluent, authoritative -- and factually wrong.

    Only the *style* half can be generated mechanically. The factual error has to
    be authored, so it is a required argument rather than something this helper
    invents; the probes shipped in ``data/test_suites/adversarial_probes.json``
    are hand-authored for the same reason. A judge that scores this highly on
    correctness has been fooled by register, which is the sycophancy failure.
    """
    body = CONFIDENT_FRAMING + false_claim + CONFIDENT_CODA
    return base_case.model_copy(
        update={
            "case_id": f"{base_case.case_id}{suffix}",
            "model_output": body,
            "tags": [*base_case.tags, "probe:confidently_wrong"],
        }
    )


def merge_accounting(items: list[CallAccounting]) -> CallAccounting:
    total = CallAccounting()
    for item in items:
        total = total.merged(item)
    return total
