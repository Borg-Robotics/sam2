"""Small shared helpers (formatting and scoring ramps)."""


def _long_axis_angle_deg(w_px, h_px, angle_deg):
    """Normalize a cv2.minAreaRect angle to the rect's LONG axis.

    cv2.minAreaRect reports the angle of the *width* (w_px) side. Dimensions are
    assigned with length = max(w_px, h_px), so when the width is the short side
    the reported angle is 90 deg off from the length axis. Add 90 deg in that
    case so angle_deg always describes the long-axis direction, then wrap into
    (-90, 90]. Consumers (ROS pose yaw, debug overlays) can rely on angle_deg
    meaning the same thing as `length`.
    """
    if h_px > w_px:
        angle_deg += 90.0
    # Wrap into (-90, 90]; a rectangle's long-axis direction is 180-periodic.
    angle_deg = (angle_deg + 90.0) % 180.0 - 90.0
    return float(angle_deg)


def long_axis_angle_diff_deg(angle_a, angle_b):
    """Smallest angle between two long-axis directions (180-periodic).

    Both inputs must ALREADY be normalized by _long_axis_angle_deg -- i.e. the
    `angle_deg` field of get_rotated_box_from_mask(). Do not re-apply the
    h_px > w_px -> +90 deg swap before calling this: the standalone scripts nest
    their own normalized_rect_angle() only because their
    get_rotated_box_from_mask returns the raw cv2.minAreaRect angle.
    """
    diff = abs(float(angle_a) - float(angle_b)) % 180.0
    return float(min(diff, 180.0 - diff))


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
