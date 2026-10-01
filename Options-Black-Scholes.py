"""
Options-Black-Scholes.py
=========================
STANDALONE reference script -- NOT used by the dashboard (which prices
options on FUTURES via Black-76, in greeks_engine.py, not spot-based
Black-Scholes). Kept here only as a simple, interactive, single-option
reference calculator for the spot case.

Patched the identical class of bugs found in Options-Black76.py (same
missing-else pattern, same lack of T<=0/sigma<=0 guards in Greeks()) --
see that file's docstring for the full explanation. Fixed the same way:
validate option_type up front, return the intrinsic-value limit instead
of dividing by zero when T<=0 or sigma<=0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

S           = float(input("Underlying Price (S): "))
K           = float(input("Strike Price (K): "))
days        = float(input("Days to Maturity: "))
r           = float(input("Risk-free Rate (%): ")) / 100
sigma       = float(input("Volatility (%): ")) / 100
option_type = input("Option type (call/put): ").strip().lower()

if option_type not in ("call", "put"):
    raise ValueError(f"Option type must be 'call' or 'put', got {option_type!r}")

T = days / 365


def BS(S, K, T, r, sigma, option_type='call'):
    if option_type not in ('call', 'put'):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)

    if sigma <= 0:
        discount = np.exp(-r * T)
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        return discount * intrinsic

    d1 = (np.log(S / K) + (r + ((sigma ** 2) / 2)) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def Greeks(S, K, T, r, sigma, option_type='call'):
    if option_type not in ('call', 'put'):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    if T <= 0 or sigma <= 0:
        if option_type == 'call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {'Delta': delta, 'Gamma': 0.0, 'Vega': 0.0, 'Theta': 0.0, 'Rho': 0.0}

    d1 = (np.log(S / K) + (r + ((sigma ** 2) / 2)) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        Delta = norm.cdf(d1)
    else:
        Delta = norm.cdf(d1) - 1

    Gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    Vega = S * np.sqrt(T) * norm.pdf(d1)

    if option_type == 'call':
        Theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
        Rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        Theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        Rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return {
        'Delta': Delta,
        'Gamma': Gamma,
        'Vega': Vega,
        'Theta': Theta,
        'Rho': Rho,
    }


print(f"\n{option_type.capitalize()} price (Black-Scholes): {BS(S, K, T, r, sigma, option_type):.4f}\n")

# range of underlying price -- centered on S, which is already the right
# convention here (unlike the futures version, S IS the reference price
# this script is built around, not a strike)
S_range = np.linspace(S * 0.5, S * 1.5, 200)
deltas, gammas, vegas, thetas, rhos = [], [], [], [], []

for s in S_range:
    g = Greeks(s, K, T, r, sigma, option_type)
    deltas.append(g['Delta'])
    gammas.append(g['Gamma'])
    vegas.append(g['Vega'])
    thetas.append(g['Theta'])
    rhos.append(g['Rho'])

# Plot Greeks vs S
fig, axes = plt.subplots(2, 3, figsize=(14, 7))

axes[0, 0].plot(S_range, deltas)
axes[0, 0].set_title("Delta")

axes[0, 1].plot(S_range, gammas)
axes[0, 1].set_title("Gamma")

axes[0, 2].plot(S_range, vegas)
axes[0, 2].set_title("Vega (per 1.00 change in sigma)")

axes[1, 0].plot(S_range, thetas)
axes[1, 0].set_title("Theta")

axes[1, 1].plot(S_range, rhos)
axes[1, 1].set_title("Rho")

axes[1, 2].axis('off')  # 6th cell is empty, hide it

for ax in axes.flat:
    ax.set_xlabel("Underlying S")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(S, color='red', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.show()
