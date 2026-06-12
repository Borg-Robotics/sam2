"""Small shared helpers (formatting and scoring ramps)."""


def json_number(value):
    if value is None:
        return None
    value = round(float(value), 3)
    if value.is_integer():
        return int(value)
    return value


def fmt3(value):
    if value is None:
        return "None"
    value = round(float(value), 3)
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def score_from_range(value, good_value, bad_value, reverse=False):
    if value is None:
        return 0.5

    value = float(value)

    if not reverse:
        if value <= good_value:
            return 1.0
        if value >= bad_value:
            return 0.0
        return 1.0 - ((value - good_value) / (bad_value - good_value))

    if value >= good_value:
        return 1.0
    if value <= bad_value:
        return 0.0
    return (value - bad_value) / (good_value - bad_value)


def score_from_bad_good(value, bad_value, good_value):
    if value <= bad_value:
        return 0.0
    if value >= good_value:
        return 1.0
    return float((value - bad_value) / (good_value - bad_value))
