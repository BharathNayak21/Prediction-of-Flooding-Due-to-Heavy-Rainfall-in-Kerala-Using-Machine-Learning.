import pandas as pd

# Load prepared Kerala ML data
df = pd.read_csv("data/kerala_ml_data.csv")

# Sort by year
df = df.sort_values("YEAR")

# Calculate 80% split point
split_index = int(len(df) * 0.8)

# Split into training and testing data
train_data = df.iloc[:split_index]
test_data = df.iloc[split_index:]

# Save the files
train_data.to_csv("data/train.csv", index=False)
test_data.to_csv("data/test.csv", index=False)

# Display information
print("Data separation completed!")

print("\nTraining data:")
print("Rows:", len(train_data))
print("Years:", train_data["YEAR"].min(), "to", train_data["YEAR"].max())

print("\nTesting data:")
print("Rows:", len(test_data))
print("Years:", test_data["YEAR"].min(), "to", test_data["YEAR"].max())