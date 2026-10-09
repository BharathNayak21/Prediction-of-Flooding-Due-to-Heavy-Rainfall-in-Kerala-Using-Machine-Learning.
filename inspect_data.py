import pandas as pd

df = pd.read_csv("data/rainfall in india 1901-2015.csv")

print("COLUMN NAMES:")
print(df.columns.tolist())

print("\nDATA SHAPE:")
print(df.shape)

print("\nFIRST 5 ROWS:")
print(df.head())