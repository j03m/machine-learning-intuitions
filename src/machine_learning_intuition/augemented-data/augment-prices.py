import numpy as np

# Example short-term data: daily prices for 14 days
short_term_prices = np.array([100, 102, 101, 103, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112])
# Example long-term data: weekly prices for 2 weeks
long_term_prices = np.array([104, 108])

# Example predictions (same structure as prices)
short_term_predictions = np.array([106, 107, 108, 109, 110])
long_term_predictions = np.array([112, 115])

def resample_to_weekly(short_term_prices, days_per_week=7):
    return np.array([short_term_prices[i:i + days_per_week].mean() for i in range(0, len(short_term_prices), days_per_week)])

def augment_long_term_with_short_term(long_term_prices, short_term_weekly_prices, weight=0.5):
    return long_term_prices * (1 - weight) + short_term_weekly_prices * weight

# Resampled short-term prices to weekly
short_term_weekly_prices = resample_to_weekly(short_term_prices)
print("Resampled Short-Term Prices to Weekly:", short_term_weekly_prices)

# Augmented long-term prices
augmented_long_term_prices = augment_long_term_with_short_term(long_term_prices, short_term_weekly_prices)
print("Augmented Long-Term Prices:", augmented_long_term_prices)
