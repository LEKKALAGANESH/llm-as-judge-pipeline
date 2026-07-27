"""LLM-as-judge evaluation pipeline.

Module layout mirrors the dependency direction, which is enforced rather than
documented: ``schema`` and ``errors`` are leaves with no internal imports;
``judge`` never imports ``aggregator`` or ``validator``; ``aggregator`` is a pure
reduction with no I/O; ``runner`` is the only module that both calls the judge
and touches the filesystem for verdicts.
"""

__version__ = "1.0.0"
