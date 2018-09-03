# --Using Apriori for Asoociation Learning--


# 1. Importing the libraries--

import numpy as np
#For Numerical calculations

import matplotlib.pyplot as plt
# For Data Vizualization

import pandas as pd
# For Data Management

# 2. Data Preprocessing--

dataset = pd.read_csv('Market_Basket.csv', header = None)
transactions = []

for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# 3. Training Apriori on the dataset--

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# 4. Visualising the results--

results = list(rules)