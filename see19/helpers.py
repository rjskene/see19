def accept_string_or_list(value):
    if not isinstance(value, str) and not isinstance(value, list):
        raise AttributeError('value {} must be type str or list'.format(value))
    else:
        return [value] if isinstance(value, str) else value