# For the NMT sampling (whether to sample, sampling interval, etc)
# See: https://github.com/chen2hang/INT_NonparametricTeaching/blob/main/src/strategy.py
def strategy_factory(strategy_type):
    if strategy_type == "simple_curriculum":
        return simple_curriculum
    elif strategy_type == "dense":
        return dense
    elif strategy_type == "dense2":
        return dense2
    else:
        raise NotImplementedError


def dense(step, max_steps):
    return True, 1


def dense2(step, max_steps, interval=10):
    if step % interval == 0:
        return True, interval
    else:
        return False, interval


"""
Curriculum learning strategy.
Intuitively, at the start of training, the model is undertrained and undergoes significant changes,
which calls for more frequent selection by the teacher but with small subsets to help the learner better digest the provided graphs;
in contrast, by the end of training, the model stabilizes and is able to digest large subsets.
"""
def simple_curriculum(step, max_steps, min_interval=5, max_interval=50, startup_ratio=0.25):
    # Global static variable to record the next trigger step
    if step == 0:
        global next_trigger_step
        next_trigger_step = 0

    # Calculate the number of steps in the startup phase
    startup_step = int(max_steps * startup_ratio)

    # Dynamically compute the increment size
    increment_size = min_interval + int(
        (max_interval - min_interval) * (step / max_steps))

    # Startup phase: frequent triggers
    if step < startup_step:
        return True, increment_size

    # Determine if the current step reaches the trigger point
    if step >= next_trigger_step:
        next_trigger_step += increment_size  # Update the next trigger step
        return True, increment_size
    else:
        return False, increment_size
