"""Compare W&B JSON metadata, allowing only one float representation step."""

import inspect
import math


def compare_metadata(actual, expected, path=''):
    differences = []
    if type(actual) is not type(expected):
        assert type(actual) in (int, float) and type(expected) in (int, float), f'Metadata type mismatch: {path}'
        assert actual == expected, f'Metadata value mismatch: {path}'
        differences.append(dict(path=path, absolute_difference=0,
                                representation_change='equal integer/float'))
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f'Metadata keys mismatch: {path}'
        for key in sorted(expected):
            differences.extend(compare_metadata(actual[key], expected[key], path+'/'+key))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f'Metadata length mismatch: {path}'
        for index, (left, right) in enumerate(zip(actual, expected)):
            differences.extend(compare_metadata(left, right, path+f'/{index}'))
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isfinite(expected), f'Nonfinite metadata: {path}'
        error = abs(actual-expected)
        assert error <= max(math.ulp(actual), math.ulp(expected)), f'Metadata value mismatch: {path}'
        if error:
            differences.append(dict(path=path, absolute_difference=error))
    else:
        assert actual == expected, f'Metadata value mismatch: {path}'
    return differences


REMOTE_COMPARISON = 'import math\n'+inspect.getsource(compare_metadata)+'\n'
