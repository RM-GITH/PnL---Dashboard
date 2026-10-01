import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import streamlit as st

F           = float(input("Forward Price (F): "))
K           = float(input("Strike Price (K): "))
days        = float(input("Days to Maturity: "))
r           = float(input("Risk-free Rate (%): ")) / 100
sigma       = float(input("Volatility (%): ")) / 100
option_type = input("Option type (call/put): ").strip().lower()

T = days / 365


def Black76(F, K, T, r, sigma, option_type='call'):

    if T <= 0:
        if option_type == 'call':
            return max(F - K, 0)
        elif option_type == 'put':
            return max(K - F, 0)

    if sigma <= 0:
        if option_type == 'call':
            return np.exp(-r * T) * max(F - K, 0)
        elif option_type == 'put':
            return np.exp(-r * T) * max(K - F, 0)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == 'put':
        return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def Greeks(F, K, T, r, sigma, option_type='call'):

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    discount = np.exp(-r * T)

    if option_type == 'call':
        Delta = discount * norm.cdf(d1)
    elif option_type == 'put':
        Delta = discount * (norm.cdf(d1) - 1)

    Gamma = discount * norm.pdf(d1) / (F * sigma * np.sqrt(T))

    Vega = discount * F * np.sqrt(T) * norm.pdf(d1)

    if option_type == 'call':
        Theta = (-discount * F * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * discount * (F * norm.cdf(d1) - K * norm.cdf(d2)))
    elif option_type == 'put':
        Theta = (-discount * F * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))

    if option_type == 'call':
        Rho = -T * discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == 'put':
        Rho = -T * discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return {
        'Delta': Delta,
        'Gamma': Gamma,
        'Vega': Vega,
        'Theta': Theta,
        'Rho': Rho
    }


# range of forward price
K_range = np.linspace(K * 0.5, K * 1.5, 200)

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
axes[0, 2].set_title("Vega")

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