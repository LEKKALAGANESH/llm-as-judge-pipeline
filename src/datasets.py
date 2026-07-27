"""Typed loaders for every on-disk dataset: suites, probes and gold labels.

Everything is validated by Pydantic on load and the pipeline refuses to run on a
malformed file, naming the case index and field that failed. A suite that half
loads is worse than one that does not load: it produces a report over a silently
truncated sample.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import SuiteLoadError
from .schema import ComparisonSuite, TestSuite

# --------------------------------------------------------------------------
# Gold labels
# --------------------------------------------------------------------------


class LabelingMethod(BaseModel):
    """Disclosed in the file itself, not only in the README, so the disclosure
    travels with the data."""

    model_config = ConfigDict(extra="forbid")

    labeler: str
    labeled_at: str
    procedure: str
    labeled_before_judge_run: bool
    relabel_performed: bool = False
    relabel_at: str | None = None
    caveats: list[str] = Field(default_factory=list)


class GoldLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    overall_score: int = Field(ge=1, le=5)
    criterion_scores: dict[str, int] = Field(default_factory=dict)
    overall_score_relabel: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Second, blind labelling pass after a washout. Enables intra-rater kappa, "
        "which is the ceiling on any judge-human agreement number. Null when not performed.",
    )
    note: str = ""


class GoldLabelSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labeling_method: LabelingMethod
    labels: list[GoldLabel]

    @property
    def by_case(self) -> dict[str, GoldLabel]:
        return {label.case_id: label for label in self.labels}


# --------------------------------------------------------------------------
# Adversarial probes
# --------------------------------------------------------------------------

ProbeKind = Literal["pointwise_threshold", "relational_pair", "identical_pair"]
ProbeRelation = Literal["variant_not_higher", "variant_higher"]


class ThresholdAssertion(BaseModel):
    """A pre-registered numeric bound. Registering the bound in the data file
    rather than deciding it after seeing results is what makes "the judge passed
    this probe" a test rather than a narrative."""

    model_config = ConfigDict(extra="forbid")

    criterion: str | None = Field(
        default=None, description="Null means the assertion is on overall_score."
    )
    max_score: int | None = Field(default=None, ge=1, le=5)
    min_score: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> ThresholdAssertion:
        if self.max_score is None and self.min_score is None:
            raise ValueError("a threshold assertion needs max_score, min_score, or both")
        return self


class Probe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_id: str
    category: str
    kind: ProbeKind
    rationale: str = ""
    input: str
    system_prompt: str = ""
    expected_output: str | None = None

    # pointwise_threshold
    model_output: str | None = None
    assertions: list[ThresholdAssertion] = Field(default_factory=list)

    # relational_pair
    baseline_output: str | None = None
    variant_output: str | None = None
    relation: ProbeRelation | None = None
    scored_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria the pair is stipulated to be identical on. The verbosity pair is "
        "scored on correctness and faithfulness only, because a padded answer legitimately "
        "differs on completeness, instruction-following and tone -- scoring all six would "
        "confound the bias signal with a real quality difference.",
    )

    # identical_pair
    output: str | None = None

    @model_validator(mode="after")
    def _kind_fields_present(self) -> Probe:
        if self.kind == "pointwise_threshold":
            if not self.model_output:
                raise ValueError(f"{self.probe_id}: pointwise_threshold needs model_output")
            if not self.assertions:
                raise ValueError(f"{self.probe_id}: pointwise_threshold needs assertions")
        elif self.kind == "relational_pair":
            if not (self.baseline_output and self.variant_output):
                raise ValueError(
                    f"{self.probe_id}: relational_pair needs baseline_output and variant_output"
                )
            if self.relation is None:
                raise ValueError(f"{self.probe_id}: relational_pair needs a relation")
            if not self.scored_criteria:
                raise ValueError(
                    f"{self.probe_id}: relational_pair needs scored_criteria (the criteria the "
                    "pair is stipulated identical on)"
                )
        elif self.kind == "identical_pair" and not self.output:
            raise ValueError(f"{self.probe_id}: identical_pair needs output")
        return self


class ProbeSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_id: str
    description: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)
    probes: list[Probe]

    @property
    def categories(self) -> list[str]:
        seen: list[str] = []
        for p in self.probes:
            if p.category not in seen:
                seen.append(p.category)
        return seen


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def _read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_file():
        raise SuiteLoadError(f"file not found: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SuiteLoadError(f"{p} is not valid JSON: {exc}") from exc


def _validate(model: type[BaseModel], data: Any, path: str | Path) -> Any:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise SuiteLoadError(f"{path} failed schema validation:\n{exc}") from exc


def load_test_suite(path: str | Path) -> TestSuite:
    suite = _validate(TestSuite, _read_json(path), path)
    if not suite.cases:
        raise SuiteLoadError(f"{path} contains zero cases; refusing to report over an empty suite")
    return suite


def load_comparison_suite(path: str | Path) -> ComparisonSuite:
    suite = _validate(ComparisonSuite, _read_json(path), path)
    if not suite.cases:
        raise SuiteLoadError(f"{path} contains zero cases; refusing to report over an empty suite")
    return suite


def load_gold_labels(path: str | Path) -> GoldLabelSet:
    gold = _validate(GoldLabelSet, _read_json(path), path)
    if not gold.labels:
        raise SuiteLoadError(
            f"{path} contains zero gold labels. Judge validation cannot proceed and still report "
            "an agreement number -- that would be a fabricated result, not a missing one."
        )
    return gold


def load_probes(path: str | Path) -> ProbeSuite:
    suite = _validate(ProbeSuite, _read_json(path), path)
    if not suite.probes:
        raise SuiteLoadError(f"{path} contains zero probes")
    return suite
