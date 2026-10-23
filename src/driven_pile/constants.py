"""Physical constants and the per-foot reason codes shared across modules."""

GAMMA_WATER = 0.0624
"""Unit weight of water, ksf/ft."""

P_ATM = 2.116
"""Atmospheric pressure, ksf."""

G_FT_S2 = 32.2
"""Gravitational acceleration, ft/s^2."""

FOOT_INCREMENT = 1.0
"""Length of one analysis increment, ft."""

OK = "ok"
ABOVE_CEILING = "above_ceiling"
BELOW_FLOOR = "below_floor"
NONFINITE = "nonfinite"
TOP_SEGMENT = "top_segment"

LIQ_BEFORE = "before"
LIQ_DURING = "during"
LIQ_AFTER = "after"
LIQ_NEVER = ""
