import pandas as pd # type: ignore
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Load dataset
df = pd.read_csv("Rotten Tomatoes Movies.csv")  # Adjust filename if needed

# Convert date column and extract year
df['in_theaters_date'] = pd.to_datetime(df['in_theaters_date'], errors='coerce')
df['year'] = df['in_theaters_date'].dt.year

# ========== ANALYSIS & VISUALIZATION ==========

color = 'skyblue'

# 1. Rating vs Tomatometer Rating → Visualization
plt.figure(figsize=(10, 6))
plt.boxplot(x=[df[df['rating'] == r]['tomatometer_rating'].dropna() for r in df['rating'].unique()],
            labels=df['rating'].unique(), patch_artist=True, boxprops=dict(facecolor=color))
plt.title("Distribution of Tomatometer Ratings by Movie Rating")
plt.xlabel("Movie Rating")
plt.ylabel("Tomatometer Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 2. Studio with Most Movies per Rating → Analysis + Visualization
studio_rating_count = df.groupby(['studio_name', 'rating']).size().reset_index(name='movie_count')
top_studio_rating = studio_rating_count.sort_values(by='movie_count', ascending=False).head(10)

print("\n🏢 Top 10 Studio-Rating Combinations by Movie Count:")
print(top_studio_rating)

# Visualization
plt.figure(figsize=(12, 6))
for rating in top_studio_rating['rating'].unique():
    subset = top_studio_rating[top_studio_rating['rating'] == rating]
    plt.bar(x=subset['studio_name'], height=subset['movie_count'], label=rating, alpha=0.7)
plt.title("Top Studios with Most Rated Movies")
plt.xlabel("Studio Name")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Rating")
plt.tight_layout()
plt.show()

# 3. Movie Rating Count Over the Years → Visualization
rating_years = df.groupby(['year', 'rating']).size().reset_index(name='count')

plt.figure(figsize=(12, 6))
for rating in rating_years['rating'].unique():
    subset = rating_years[rating_years['rating'] == rating]
    plt.plot(subset['year'], subset['count'], marker='o', label=rating)
plt.title("Number of Movies Released per Rating Over the Years")
plt.xlabel("Year")
plt.ylabel("Movie Count")
plt.legend(title="Rating")
plt.tight_layout()
plt.show()