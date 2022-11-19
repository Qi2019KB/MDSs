import numpy as np


def update_ema_variables(model, ema_model, args):
    global_step = args.global_step + 1
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), args.ema_decay)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    return global_step


# min --> max
def consWeight_increase(epo, args):
    return args.consWeight_max * _sigmoid_rampup(epo, args.consWeight_rampup)


# min --> max
def pseudoWeight_increase(epo, args):
    return args.pseudoWeight_max * _sigmoid_rampup(epo, args.pseudoWeight_rampup)


# min --> max
def PALWeight_increase(epo, args):
    return args.PALWeight_max * _sigmoid_rampup(epo, args.PALWeight_rampup)


# min --> max
def uncPCT_increase(epo, args):
    return args.uncPCT_max * _sigmoid_rampup(epo, args.uncPCT_rampup)


# min --> max
def scoreThr_increase(epo, args):
    return args.scoreThr_min + (args.scoreThr_max - args.scoreThr_min) * _sigmoid_rampup(epo, args.scoreThr_rampup)


# max --> min
def PALWeight_decrease(epo, args):
    return args.PALWeight_max * (1.0 - _sigmoid_rampup(epo, args.PALWeight_rampup))


# max --> min
def uncPCT_decrease(epo, args):
    return args.uncPCT_max * (1.0 - _sigmoid_rampup(epo, args.uncPCT_rampup))


def _sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))