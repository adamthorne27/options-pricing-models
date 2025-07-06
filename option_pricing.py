import math
import numpy as np
from enum import Enum
from scipy.stats import norm

class OptionType(Enum):
    Call = "Call"
    Put = "Put"

class EuropeanOption:
    def __init__(self, S, K, T, r, sigma, option_type):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type

class BlackScholesModel:
    def price(self, option):
        S, K, T, r, sigma = option.S, option.K, option.T, option.r, option.sigma
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option.option_type == OptionType.Call:
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

class MonteCarloModel:
    def __init__(self, N):
        self.N = N

    def price(self, option):
        S, K, T, r, sigma = option.S, option.K, option.T, option.r, option.sigma
        Z = np.random.normal(0, 1, self.N)
        ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        if option.option_type == OptionType.Call:
            payoff = np.maximum(ST - K, 0)
        else:
            payoff = np.maximum(K - ST, 0)
        return np.exp(-r * T) * np.mean(payoff)

class BinomialModel:
    def __init__(self, steps):
        self.steps = steps

    def price(self, option):
        S, K, T, r, sigma = option.S, option.K, option.T, option.r, option.sigma
        dt = T / self.steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        p = (math.exp(r * dt) - d) / (u - d)
        discount = math.exp(-r * dt)

        prices = [S * (u ** j) * (d ** (self.steps - j)) for j in range(self.steps + 1)]
        if option.option_type == OptionType.Call:
            values = [max(0, price - K) for price in prices]
        else:
            values = [max(0, K - price) for price in prices]

        for i in range(self.steps - 1, -1, -1):
            values = [discount * (p * values[j + 1] + (1 - p) * values[j]) for j in range(i + 1)]

        return values[0]

class Delta:
    def evaluate(self, option):
        S, K, T, r, sigma = option.S, option.K, option.T, option.r, option.sigma
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        if option.option_type == OptionType.Call:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

class Gamma:
    def evaluate(self, option):
        S, K, T, r, sigma = option.S, option.K, option.T, option.r, option.sigma
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))  

class Theta:
    def evaluate(self, option):
        S, K, T, r, sigma = option.S, option.K, option.T, option.r, option.sigma
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        first_term = - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))

        if option.option_type == OptionType.Call:
            return first_term - r * K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return first_term + r * K * math.exp(-r * T) * norm.cdf(-d2)

class Vega:
    def evaluate(self, option):
        S, K, T, r, sigma = option.S, option.K, option.T, option.r, option.sigma
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return S * norm.pdf(d1) * math.sqrt(T) / 100  

class Rho:
    def evaluate(self, option):
        S, K, T, r, sigma = option.S, option.K, option.T, option.r, option.sigma
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option.option_type == OptionType.Call:
            return K * T * math.exp(-r * T) * norm.cdf(d2) / 100  
        else:
            return -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
