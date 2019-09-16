from miraiml.util import sample_random_len, is_valid_filename


def test_sample_random_len():
    if len(sample_random_len([])) != 0:
        raise AssertionError()

    lst = range(10)
    result = sample_random_len(lst)

    if len(result) > len(lst):
        raise AssertionError()

    if len(set(result) - set(lst)) > 0:
        raise AssertionError()


def test_is_valid_filename():
    invalid_names = ['name*', 'a%', '$a', '', '..name', 'log/log.ext', ' ', '.']
    valid_names = ['log', 'lo(g)_log', '(', '_', '-', '.a']

    for name in invalid_names:
        if is_valid_filename(name):
            raise AssertionError()

    for name in valid_names:
        if not is_valid_filename(name):
            raise AssertionError()
