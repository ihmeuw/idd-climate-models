"""
structures/__init__.py
The four model structures. Each assembles distributions into a multi-stage model.

  single        — E[Y | X] directly from one distribution
  hurdle        — P(Y>0) * E[Y | Y>0]  (2 stages)
  pot           — binary + bulk + GPD tail  (3 stages, fixed p_exceed)
  double_hurdle — binary + covariate-dependent binary + bulk + tail  (4 stages)
"""

from . import single, hurdle, pot, double_hurdle

STRUCTURES = {
    'single':        single,
    'hurdle':        hurdle,
    'pot':           pot,
    'double_hurdle': double_hurdle,
}

__all__ = ['single', 'hurdle', 'pot', 'double_hurdle', 'STRUCTURES']
