pip install pandas matplotlib seaborn scikit-learn

import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print("❌ Error loading dataset:", e)

# Display the first few rows
print("\n🔍 First 5 Rows of the Dataset:")
print(df.head())

# Check data types and missing values
print("\n📊 Dataset Info:")
print(df.info())

print("\n🧩 Missing Values:")
print(df.isnull().sum())

# Clean dataset (if missing values existed)
df = df.dropna()

# Basic statistics
print("\n📉 Descriptive Statistics:")
print(df.describe())

# Group by species and compute mean of numerical columns
grouped = df.groupby("target").mean()
print("\n🔎 Mean values per species:")
print(grouped)

# Map numeric target to species name for better readability
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

# Re-run grouped stats with species name
print("\n🔍 Mean values per species (named):")
print(df.groupby("species").mean())
# Visualize the dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

# 1. Line Chart – Simulate time-series trend (fake index as time)
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
plt.title("Simulated Time-Series of Sepal and Petal Length")
plt.xlabel("Sample Index (as Time)")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart – Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x="species", y="petal length (cm)", data=df, ci=None)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram – Distribution of sepal length
plt.figure(figsize=(6, 4))
sns.histplot(df["sepal length (cm)"], kde=True, bins=20, color='skyblue')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot – Sepal length vs. Petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df)
plt.title("Sepal vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

df.to_csv("cleaned_iris.csv", index=False)
print("📁 Cleaned dataset saved as 'cleaned_iris.csv'")
