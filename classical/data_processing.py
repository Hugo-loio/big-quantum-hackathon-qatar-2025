import pandas as pd
import numpy as np

def extract_prices(name):
    # Load the CSV file
    df = pd.read_csv("raw_datasets/" + str(name)) 

    # Extract only price columns (avg_price_YYYY)
    price_cols = [col for col in df.columns if col.startswith("avg_price_")]

    # Convert to NumPy
    prices = df[price_cols].to_numpy()

    # Transpose: rows = years, columns = companies
    return prices.T

def compute_returns(prices):
    # Simplest method for computing returns
    rs = (prices[1:] - prices[:-1])/prices[:-1]

    return rs 

def compute_correlations_time(rs, t):
    rs_t = rs[:t]
    rs_avg = np.average(rs_t, axis = 0)
    n = rs.shape[1]
    cov = np.empty((n,n))

    for i in range(n):
        for j in range(n):
            cov[i,j] = np.sum((rs_t[:,i] - rs_avg[i]) * (rs_t[:,j] - rs_avg[j]))/(t - 1)
    return cov


datasets = ["qatar_10_companies_2015_2025_avg_prices.csv", "qatar_54_companies_2015_2025_avg_prices.csv"]

prices_datasets = [extract_prices(name) for name in datasets]
rs_datasets = [compute_returns(prices) for prices in prices_datasets]
rs_avg_datasets = [np.average(rs, axis = 0) for rs in rs_datasets]
covs_datasets = [compute_correlations_time(rs, len(rs)) for rs in rs_datasets]

for i,name in enumerate(datasets):
    np.savetxt("processed_datasets/returns_" + str(i+1) + ".csv", rs_avg_datasets[i], delimiter=',')
    np.savetxt("processed_datasets/cov_" + str(i+1) + ".csv", covs_datasets[i], delimiter=',')
