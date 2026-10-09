import pandas as pd

# Load Kerala data
df = pd.read_csv("data/kerala_rainfall.csv")

# Create June change
df["June_change"] = abs(df["MAY"] - df["JUN"])

# Create Flood Risk target
KERALA_THRESHOLD = 2049.3

df["Flood_Risk"] = (df["Jun-Sep"] > KERALA_THRESHOLD).astype(int)

# Keep only the columns needed for our ML project
ml_data = df[
    ["YEAR", "Mar-May", "JUN", "June_change", "Flood_Risk"]
]

# Rename JUN to June
ml_data = ml_data.rename(columns={
    "JUN": "June"
})

# Sort by year
ml_data = ml_data.sort_values("YEAR")

# Save prepared dataset
ml_data.to_csv("data/kerala_ml_data.csv", index=False)

# Display results
print("ML dataset created successfully!")

print("\nFirst 10 rows:")
print(ml_data.head(10))

print("\nDataset shape:")
print(ml_data.shape)

print("\nFlood Risk counts:")
print(ml_data["Flood_Risk"].value_counts())