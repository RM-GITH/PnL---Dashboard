import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import streamlit as st


S           = float(input("Underlying Price (S): "))
K           = float(input("Strike Price (K): "))
days        = float(input("Days to Maturity: "))
r           = float(input("Risk-free Rate (%): ")) / 100
sigma       = float(input("Volatility (%): ")) / 100
option_type = input("Option type (call/put): ").strip().lower()

T = days / 365

def BS(S,K,T,r,sigma,option_type='call'):

    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        elif option_type == 'put':
            return max(K - S, 0)

    if sigma <= 0:
        if option_type == 'call':
            return max(S * np.exp(-r * T) - K * np.exp(-r * T), 0)
        elif option_type == 'put':
            return max(K * np.exp(-r * T) - S * np.exp(-r * T), 0)

    d1 = (np.log(S / K) + (r + ((sigma ** 2) / 2)) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def Greeks(S,K,T,r,sigma,option_type='call'):



    d1 = (np.log(S / K) + (r + ((sigma ** 2) / 2)) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        Delta = norm.cdf(d1)
    elif option_type == 'put':
        Delta = norm.cdf(d1) - 1

    Gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    Vega = S * np.sqrt(T) * norm.pdf(d1)

    if option_type == 'call':
        Theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == 'put':
        Theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + r * K * np.exp(-r*T) * norm.cdf(-d2)

    if option_type == 'call':
        Rho = K * T * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == 'put':
        Rho = -K * T * np.exp(-r*T) * norm.cdf(-d2)

    return {
            'Delta': Delta,
            'Gamma': Gamma,
            'Vega': Vega,
            'Theta': Theta,
            'Rho': Rho

        }

# range of underlying price
S_range = np.linspace(S * 0.5, S * 1.5, 200)
deltas = []
gammas = []
vegas = []
thetas = []
rhos = []

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
axes[0, 2].set_title("Vega")

axes[1, 0].plot(S_range, thetas)
axes[1, 0].set_title("Theta")

axes[1, 1].plot(S_range, rhos)
axes[1, 1].set_title("Rho")

axes[1, 2].axis('off')  # 6th cell is empty, hide it

for ax in axes.flat:
        ax.set_xlabel("Underlying S")
        ax.axhline(0, color='black', linewidth=0.5)  # zero line
        ax.axvline(S, color='red', linewidth=0.5, linestyle='--')  # current S

plt.tight_layout()
plt.show()