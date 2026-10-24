"""Task-based generation with lazy model loading."""


def run(args):
    from .runtime import run as execute

    return execute(args)


def load_for_check(*args, **kwargs):
    from .runtime import load_for_check as execute

    return execute(*args, **kwargs)


def run_smoke(*args, **kwargs):
    from .runtime import run_smoke as execute

    return execute(*args, **kwargs)
