"""Local chemistry/publication units; dispatcher has separate tests."""
def inline_dispatch(items, settings, *, on_result=None, **kwargs):
    from partiflow.postprocess.runner import _process_one
    results = []
    for index, item in enumerate(items):
        result = _process_one(item, settings)
        results.append(result)
        if on_result is not None:
            on_result(index, result)
    return results, []
