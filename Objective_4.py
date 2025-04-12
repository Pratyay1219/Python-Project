import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Load dataset
df = pd.read_csv("Rotten Tomatoes Movies.csv") # Update path as needed

# Clean and prepare data
df = df.dropna(subset=['runtime_in_minutes'])
df['runtime_in_minutes'] = pd.to_numeric(df['runtime_in_minutes'], errors='coerce')

color = 'skyblue'

# ========== 1. Histogram of All Movie Runtimes ==========
df = df[df['runtime_in_minutes'] != df['runtime_in_minutes'].max()]
plt.figure(figsize=(10, 6))
sns.histplot(df['runtime_in_minutes'], bins=30, color='skyblue', kde=True)
plt.title("Histogram of Movie Runtimes")
plt.xlabel("Runtime (in minutes)")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.show()

# ========== 2. Violin Plot of Rating-wise Runtime ==========
plt.figure(figsize=(12, 6))
sns.boxplot(x='rating', y='runtime_in_minutes', data=df, palette='Set3')
plt.title("Runtime Distribution by Movie Rating (Box Plot)")
plt.xlabel("Movie Rating")
plt.ylabel("Runtime (in minutes)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========== 3. Buckets of Runtime ==========
bins = [0, 90, 120, 150, df['runtime_in_minutes'].max()]
labels = ['<90 min', '90–120 min', '120–150 min', '>150 min']
df['runtime_bucket'] = pd.cut(df['runtime_in_minutes'], bins=bins, labels=labels, include_lowest=True)

# Count and average ratings
runtime_summary = df.groupby('runtime_bucket').agg({
'movie_title': 'count',
'tomatometer_rating': 'mean',
'audience_rating': 'mean'
}).rename(columns={'movie_title': 'movie_count'}).reset_index()

print("\n🎬 Runtime Buckets Summary:")
print(runtime_summary)

# Visualization
plt.figure(figsize=(8, 8))
plt.pie(runtime_summary['movie_count'], 
        labels=runtime_summary['runtime_bucket'], 
        autopct='%1.1f%%', 
        colors=sns.color_palette("Set3"),
        startangle=140)
plt.title("Movie Count Distribution by Runtime Bucket")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(runtime_summary['runtime_bucket'], runtime_summary['tomatometer_rating'], label='Critic Rating', marker='o', color=color)
plt.plot(runtime_summary['runtime_bucket'], runtime_summary['audience_rating'], label='Audience Rating', marker='s', linestyle='--', color='lightcoral')
plt.title("Average Ratings by Runtime Bucket")
plt.xlabel("Runtime Bucket")
plt.ylabel("Average Rating")
plt.legend()
plt.tight_layout()
plt.show()

# ========== 4. Highest Rated Movies by Runtime ==========
# Audience
audience_top = df[['movie_title', 'runtime_in_minutes', 'audience_rating']].dropna().sort_values(by='audience_rating', ascending=False).head(10)
# Critic
critic_top = df[['movie_title', 'runtime_in_minutes', 'tomatometer_rating']].dropna().sort_values(by='tomatometer_rating', ascending=False).head(10)

print("\n🎯 Top 10 Highest Rated Movies by Audience:")
print(audience_top)

avg_runtime_audience = audience_top['runtime_in_minutes'].mean()
print(f"\n📌 Average Runtime of Top 10 Audience-Rated Movies: {avg_runtime_audience:.2f} minutes")

print("\n🎯 Top 10 Highest Rated Movies by Critic:")
print(critic_top)

avg_runtime_critic = critic_top['runtime_in_minutes'].mean()
print(f"📌 Average Runtime of Top 10 Critic-Rated Movies: {avg_runtime_critic:.2f} minutes")