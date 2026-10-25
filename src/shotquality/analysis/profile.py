"""Turn two numbers into a sentence a non-modeler can read."""

from __future__ import annotations


def profile_sentence(selection_per_100: float, making_per_100: float, band: float = 2.0) -> str:
    """Describe shot selection and shot making.

    Both inputs are points per 100 shots relative to average (selection) or
    relative to expectation (making). Gaps smaller than ``band`` are called
    average so a 0.4-point wobble is not narrated as a skill.
    """
    if band < 0:
        raise ValueError("band cannot be negative")
    sel_high = selection_per_100 > band
    sel_low = selection_per_100 < -band
    mak_high = making_per_100 > band
    mak_low = making_per_100 < -band

    if sel_high and mak_high:
        return (
            "Takes higher-value shots than the league and makes more of them than those shots are worth."
        )
    if sel_high and mak_low:
        return "Gets good looks, but makes fewer of them than the quality of those looks predicts."
    if sel_low and mak_high:
        return "Shot selection is tougher than average, and finishing is better than those shots predict."
    if sel_low and mak_low:
        return "Takes lower-value shots than the league and finishes them below expectation."
    if sel_high:
        return (
            "Takes higher-value shots than the league, with about average shot making "
            "once shot quality is accounted for."
        )
    if sel_low:
        return (
            "Takes lower-value shots than the league, with about average shot making "
            "once shot quality is accounted for."
        )
    if mak_high:
        return "Shot selection is about average, and finishing is better than those shots predict."
    if mak_low:
        return "Shot selection is about average, but finishing is below what those shots predict."
    return "About average in both shot selection and shot making."
