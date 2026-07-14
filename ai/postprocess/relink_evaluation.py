def normalize_identity_ground_truth(value: str | None) -> str | None:
    """Return a non-empty identity label suitable for offline relink evaluation."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def wrong_relink_if_evaluable(
    expected_identity: str | None,
    observed_identity: str | None,
) -> bool | None:
    """Compare identity labels only when both sides have ground truth."""
    expected = normalize_identity_ground_truth(expected_identity)
    observed = normalize_identity_ground_truth(observed_identity)
    if expected is None or observed is None:
        return None
    return expected != observed
