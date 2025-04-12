import pandas as pd # type: ignore
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Load your dataset
df = pd.read_csv('Rotten Tomatoes Movies.csv')  # Change filename if needed

# ========== Clean & Prepare Date ==========
df['in_theaters_date'] = pd.to_datetime(df['in_theaters_date'], errors='coerce')
df['release_year'] = df['in_theaters_date'].dt.year

# ===================== ANALYSIS =====================

# 1. Top 10 Years with Highest Number of Movies
top10_years = df['release_year'].value_counts().sort_values(ascending=False).head(10)
print("Top 10 Years with Highest Number of Movies:\n")
print(top10_years)

# 2. Least 10 Years with Lowest Number of Movies
least10_years = df['release_year'].value_counts().sort_values(ascending=True).head(10)
print("\nLeast 10 Years with Lowest Number of Movies:\n")
print(least10_years)

# 3. Top 10 Studios with Most Movies
top10_studios = df['studio_name'].value_counts().head(10)
print("\nTop 10 Studios with Most Movies:\n")
print(top10_studios)

# 4. Movie Count by Audience Rating Bins
bins = [0, 20, 40, 60, 80, 100]
labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
df['rating_bin'] = pd.cut(df['audience_rating'], bins=bins, labels=labels, include_lowest=True)
rating_bins_count = df['rating_bin'].value_counts().sort_index()
print("\nNumber of Movies in Audience Rating Ranges:")
print(rating_bins_count)

# ===================== VISUALIZATION =====================

color = 'skyblue'

#Over the years analysis
df['year'] = df['in_theaters_date'].dt.year
df['year'].value_counts().sort_index().plot(kind='line')
plt.title('Movies Released Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.grid(True)
plt.show()

# Top 10 Studios with Most Movies
plt.figure(figsize=(10, 6))
plt.barh(y=top10_studios.index, width=top10_studios.values, color=color)
plt.title("Top 10 Studios by Number of Movies")
plt.xlabel("Number of Movies")
plt.ylabel("Studio")
plt.tight_layout()
plt.show()

# Audience Rating Bin Distribution
plt.figure(figsize=(8, 5))
plt.bar(x=rating_bins_count.index, height=rating_bins_count.values, color=color)
plt.title("Number of Movies in Audience Rating Bins")
plt.xlabel("Rating Range")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.show()