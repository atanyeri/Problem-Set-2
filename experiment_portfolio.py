"""Experiment-portfolio simulator and decision rules (ECON 5200 PS2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

P_REAL = 0.20 # share of experiments with a real effect
EFF_MEAN = 0.10 # real effects ~Normal(EFF_MEAN,EFF_SD)
EFF_SD = 0.04
N_LOW, N_HIGH = 200, 5000 # users per arm
SIGMA = 1.0 # outcome sd# outcome sd

# evaluation settings:
C = 0.03  # assumed cost per shipped experiment in sd units
NULL_BELOW = 0.02 # true effect below this counts as zero
MIN_N = 1000  # users per arm floow for the n_floor rule

RULE_NAMES = ('client', 'strict', 'shrinkage', 'n_floor', 'ship_all')


def simulate_portfolio(n: int, seed: int, **kwargs):
    """Return the experiment log with the true effect attached."""
    p_real = kwargs.get('p_real', P_REAL)
    eff_mean = kwargs.get('eff_mean', EFF_MEAN)
    eff_sd = kwargs.get('eff_sd', EFF_SD)
    n_low = kwargs.get('n_low', N_LOW)
    n_high = kwargs.get('n_high', N_HIGH)
    sigma = kwargs.get('sigma', SIGMA)

    rng = np.random.default_rng(seed)
    real = rng.random(n) < p_real
    true_eff = np.where(real, rng.normal(eff_mean, eff_sd, size=n), 0.0)
    users = rng.integers(n_low, n_high, size=n)
    se = sigma * np.sqrt(2/users)
    est = true_eff + rng.normal(0, se)

    log = pd.DataFrame({'true': true_eff, 'n': users, 'se': se, 'est': est})
    log['p'] = 2 * stats.norm.sf(np.abs(log['est'] / log['se']))

    # posterior mean of the true effect under the mixture prior
    f0 = stats.norm.pdf(log['est'], 0, log['se'])
    f1 = stats.norm.pdf(log['est'], eff_mean, np.sqrt(eff_sd**2 + log['se']**2))
    p_real = p_real * f1 / (p_real * f1 + (1 - p_real) * f0)
    w = eff_sd**2 / (eff_sd**2 + log['se']**2)
    log['post_mean'] = p_real * (eff_mean + w * (log['est'] - eff_mean))

    return log


def apply_rule(log, rule):
    """Ship decisions under `rule`: the p < 0.05 threshold and at least one other."""
    if rule == "client":
        mask = (log["p"] < 0.05) & (log["est"] > 0)
    elif rule == "strict":
        mask = (log["p"] < 0.005) & (log["est"] > 0)
    elif rule == "shrinkage":
        mask = log["post_mean"] > C
    elif rule == "n_floor":
        mask = (log["p"] < 0.05) & (log["est"] > 0) & (log["n"] >= MIN_N)
    elif rule == "ship_all":
        mask = pd.Series(True, index=log.index)
    else:
      raise ValueError(f'unknown rule {rule}')
    return mask.astype(bool)


def evaluate(log, rule):
    """Realised true return, ship count and false-ship rate under `rule`."""
    s = log[apply_rule(log, rule)]
    shipped = len(s)
    realised = s['true'].sum()
    return {'shipped': shipped,
            'realised': realised,
            'false_ship_rate': (s['true'] < NULL_BELOW).mean(),
            'mean_true': s['true'].mean(),
            'mean_est': s['est'].mean(),
            'net': realised - shipped * C
            }


if __name__ == "__main__":
    log = simulate_portfolio(200, 42)
    print(pd.DataFrame({name: evaluate(log, name) for name in RULE_NAMES}).T.round(3))

    # 1. same seed gives the same portfolio and the log carries the truth
    assert log.equals(simulate_portfolio(200, seed=42)) and len(log) == 200
    assert {'true', 'est', 'se', 'p'} <= set(log.columns)
    # 2. Stricter rules ship a subset of what the client's rule ships
    assert (apply_rule(log, 'strict') <= apply_rule(log, 'client')).all()
    assert (apply_rule(log, 'n_floor') <= apply_rule(log, 'client')).all()
    # 3. Counts rates and net are internally consistent
    ev = evaluate(log, 'client')
    assert evaluate(log, 'ship_all')['shipped'] == 200
    assert ev['shipped'] == apply_rule(log, 'client').sum()
    assert 0.0 <= ev['false_ship_rate'] <= 1.0
    assert abs(ev['net'] - (ev['realised'] - C * ev['shipped'])) < 1e-12
    # 4. An unknown rule name raises an error
    try:
        apply_rule(log, 'nonsense')
        raise AssertionError('unknown rule should raise ValueError')
    except ValueError:
        pass
    # 5. Winners curse: on a large portfolio shipped estimates exceed the truth.
    big = evaluate(simulate_portfolio(5000, seed=1), 'client')
    assert big['mean_est'] > big['mean_true']
    print("experiment_portfolio: all assertions passed")
