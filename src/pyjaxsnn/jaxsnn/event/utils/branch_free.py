import contextlib

_BRANCH_FREE = True


def set_branch_free(
    enabled: bool,
) -> bool:
    """
    Globally switch `custom_lax.cond`/`switch` between branch-free and
    real control flow.

    :param enabled: Enable or disable branch-free computation
    :returns: Returns the previous setting.
    """
    global _BRANCH_FREE  # pylint: disable=global-statement
    previous, _BRANCH_FREE = _BRANCH_FREE, enabled
    return previous


def is_branch_free() -> bool:
    return _BRANCH_FREE


@contextlib.contextmanager
def branch_free(enabled: bool = True):
    """
    Context manager to temporarily switch branch-free mode for
    `custom_lax.cond`/`switch`.
    """
    previous = set_branch_free(enabled)
    try:
        yield
    finally:
        set_branch_free(previous)
