import math

# For dynamically adjusting the sampling ratio
# See: https://github.com/chen2hang/INT_NonparametricTeaching/blob/main/src/scheduler.py
def scheduler_factory(scheduler_type):
    if scheduler_type == "step":
        return mt_step
    elif scheduler_type == "constant":
        return mt_constant
    elif scheduler_type == "linear":
        return mt_linear
    elif scheduler_type == "cosine":
        return mt_cosine_annealing
    elif scheduler_type == "reverse-cosine":
        return mt_revcosine_annealing
    else:
        raise NotImplementedError


def mt_constant(step, max_step, mt_ratio):
    return mt_ratio


def mt_linear(step, max_step, mt_ratio):
    new_ratio = mt_ratio + (step/max_step) * (1.0 - mt_ratio)
    return new_ratio


def mt_step(step, max_step, mt_ratio, n_stages=50):
    interval = max_step / n_stages
    ratio_step = (1.0 - mt_ratio) / n_stages
    stage = step // interval
    new_ratio = min(mt_ratio + stage * ratio_step, 1.0)
    return new_ratio


def mt_cosine_annealing(step, max_step, mt_ratio, max_ratio=1.0):
    new_ratio = max_ratio - 1/2 * (max_ratio - mt_ratio) * (1 + math.cos(step/max_step*math.pi))
    return new_ratio


def mt_revcosine_annealing(step, max_step, mt_ratio, min_ratio=0.2):
    new_ratio = mt_ratio - 1/2 * (mt_ratio - min_ratio) * (1 - math.cos(step/max_step*math.pi))
    return new_ratio
