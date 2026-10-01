"""
Options-Black76.py
===================
STANDALONE reference script -- NOT used by the dashboard. The integrated,
tested, actively-used version of this logic lives in greeks_engine.py
(black76_price, black76_greeks), which also adds: a strike/expiry vol
surface instead of flat vol, position-level aggregation, FIFO realized
P&L, and a documented Vega unit convention. Use that module for anything
feeding the dashboard; this file is kept only as a simple, interactive,
single-option reference calculator.

Patched bugs (previously real, reproducible defects in this file):
  - Greeks() crashed with UnboundLocalError on any option_type that wasn't
    exactly 'call'/'put', because Delta/Theta/Rho were only assigned
    inside if/elif blocks with no else.
  - Black76() failed SILENTLY on the same bad input in the normal
    (T>0, sigma>0) path -- no else branch, no final return, so it
    returned None with no error at all.
  - Greeks() had no guard for T<=0 or sigma<=0 at all (unlike Black76(),
    which at least partially handled it), so an at-expiry or zero-vol
    input would silently produce NaN/Inf via a 0/0 division, with only a
    numpy RuntimeWarning (easy to miss) rather than a clear error or a
    sensible intrinsic-value limit.
All three are fixed below the same way greeks_engine.py handles them:
validate option_type up front (raise clearly on anything else), and
return the correct intrinsic-value limit when T<=0 or sigma<=0 instead
of dividing by zero.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

F           = float(input("Forward Price (F): "))
K           = float(input("Strike Price (K): "))
days        = float(input("Days to Maturity: "))
r           = float(input("Risk-free Rate (%): ")) / 100
sigma       = float(input("Volatility (%): ")) / 100
option_type = input("Option type (call/put): ").strip().lower()

if option_type not in ("call", "put"):
    raise ValueError(f"Option type must be 'call' or 'put', got {option_type!r}")

T = days / 365


def Black76(F, K, T, r, sigma, option_type='call'):
    if option_type not in ('call', 'put'):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    if T <= 0:
        return max(F - K, 0) if option_type == 'call' else max(K - F, 0)

    if sigma <= 0:
        intrinsic = max(F - K, 0) if option_type == 'call' else max(K - F, 0)
        return np.exp(-r * T) * intrinsic

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def Greeks(F, K, T, r, sigma, option_type='call'):
    if option_type not in ('call', 'put'):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    if T <= 0 or sigma <= 0:
        # At expiry or zero vol: no time value, Greeks collapse to the
        # intrinsic-value limit. Only Delta is non-trivial in that limit.
        if option_type == 'call':
            delta = 1.0 if F > K else 0.0
        else:
            delta = -1.0 if F < K else 0.0
        return {'Delta': delta, 'Gamma': 0.0, 'Vega': 0.0, 'Theta': 0.0, 'Rho': 0.0}

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    discount = np.exp(-r * T)

    if option_type == 'call':
        Delta = discount * norm.cdf(d1)
    else:
        Delta = discount * (norm.cdf(d1) - 1)

    Gamma = discount * norm.pdf(d1) / (F * sigma * np.sqrt(T))

    Vega = discount * F * np.sqrt(T) * norm.pdf(d1)

    if option_type == 'call':
        Theta = (-discount * F * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * discount * (F * norm.cdf(d1) - K * norm.cdf(d2)))
        Rho = -T * discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        Theta = (-discount * F * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))
        Rho = -T * discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return {
        'Delta': Delta,
        'Gamma': Gamma,
        'Vega': Vega,
        'Theta': Theta,
        'Rho': Rho
    }


print(f"\n{option_type.capitalize()} price (Black-76): {Black76(F, K, T, r, sigma, option_type):.4f}\n")

# range of forward price -- centered on F (the forward), not K (the strike),
# so the plotted range stays representative even when the trade is deep
# ITM/OTM. (The original version of this script centered on K instead,
# which could push the actually-relevant region to one edge of the chart
# or off it entirely -- see greeks_engine.py's dashboard chart for the
# corrected convention this now matches.)
K_range = np.linspace(F * 0.5, F * 1.5, 200)

deltas, gammas, vegas, thetas, rhos = [], [], [], [], []

for k in K_range:
    g = Greeks(F, k, T, r, sigma, option_type)
    deltas.append(g['Delta'])
    gammas.append(g['Gamma'])
    vegas.append(g['Vega'])
    thetas.append(g['Theta'])
    rhos.append(g['Rho'])

# Plot Greeks vs K
fig, axes = plt.subplots(2, 3, figsize=(14, 7))

axes[0, 0].plot(K_range, deltas)
axes[0, 0].set_title("Delta")

axes[0, 1].plot(K_range, gammas)
axes[0, 1].set_title("Gamma")

axes[0, 2].plot(K_range, vegas)
axes[0, 2].set_title("Vega (per 1.00 change in sigma -- see greeks_engine.py for the per-vol-point convention)")

axes[1, 0].plot(K_range, thetas)
axes[1, 0].set_title("Theta")

axes[1, 1].plot(K_range, rhos)
axes[1, 1].set_title("Rho")

axes[1, 2].axis('off')

for ax in axes.flat:
    ax.set_xlabel("Strike Price K")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(F, color='red', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.show()
