import pandas as pd

# Load the full rainfall dataset
df = pd.read_csv("data/rainfall in india 1901-2015.csv")

# Select Kerala only
kerala = df[df["SUBDIVISION"] == "KERALA"]

# Save Kerala data
kerala.to_csv("data/kerala_rainfall.csv", index=False)

# Display information
print("Kerala data created successfully!")
print("Number of rows:", len(kerala))

print("\nFirst 5 Kerala rows:")
print(kerala.head())