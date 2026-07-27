"""Exception taxonomy. Leaf module: no internal imports.

The distinction that matters most here is between errors that justify a retry
and errors that do not. Retrying an authentication failure inside a per-case
loop multiplies one misconfiguration into hundreds of pointless requests and
delays the only useful signal -- which key is wrong -- by minutes.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base for every error this pipeline raises deliberately."""


class ConfigError(PipelineError):
    """Configuration is invalid. Always fatal at startup."""


class CrossFamilyViolation(ConfigError):
    """The judge shares a model family with a candidate generator.

    Hard stop, not a warning. A silently bypassable check would leave the
    self-enhancement-bias mitigation claimed but not enforced, which is worse
    than not claiming it. The deliberate same-family measurement run
    (`measure-self-enhancement`) opts out explicitly via --allow-same-family.
    """


class SuiteLoadError(PipelineError):
    """A test suite, probe file or gold-label file failed schema validation."""


class QuotaExceeded(PipelineError):
    """The per-run call or cost circuit breaker tripped.

    Raised before the offending request is issued, so the breaker bounds actual
    provider usage rather than reporting it after the fact.
    """


class FatalProviderError(PipelineError):
    """A non-transient provider failure: auth, permission, or bad request.

    Never retried. The message names which provider's credential to check.
    """


class JudgeParseError(PipelineError):
    """The judge's output could not be parsed into the verdict schema after the
    configured number of repair attempts."""


class JudgeCallFailed(PipelineError):
    """The judge API call failed after the configured transient retries."""
