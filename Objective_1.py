import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# ------------------ Data Preparation ------------------

df = pd.read_csv('Rotten Tomatoes Movies.csv')   # Replace with your CSV filename

df['audience_rating'] = pd.to_numeric(df['audience_rating'], errors='coerce')
df['tomatometer_rating'] = pd.to_numeric(df['tomatometer_rating'], errors='coerce')
df['in_theaters_date'] = pd.to_datetime(df['in_theaters_date'], errors='coerce')
df['year'] = df['in_theaters_date'].dt.year

df = df.dropna(subset=['audience_rating', 'tomatometer_rating', 'genre', 'movie_title', 'year'])

# ------------------ 🔍 ANALYSIS SECTION ------------------

# 1. Top 10 Genres by Audience and Critic
genre_avg = df.groupby('genre')[['audience_rating', 'tomatometer_rating']].mean()

top10_genre_audience = genre_avg.sort_values(by='audience_rating', ascending=False).head(10)
top10_genre_critic = genre_avg.sort_values(by='tomatometer_rating', ascending=False).head(10)

# 2. Top 10 Movies by Audience and Critic
top10_movies_audience = df.sort_values(by='audience_rating', ascending=False)[['movie_title', 'audience_rating']].head(10)
top10_movies_critic = df.sort_values(by='tomatometer_rating', ascending=False)[['movie_title', 'tomatometer_rating']].head(10)

print("\n🎬 Top 10 Movies by Audience Rating:\n", top10_movies_audience.to_string(index=False))
print("\n🎬 Top 10 Movies by Critic Rating:\n", top10_movies_critic.to_string(index=False))

# 3. Top 10 Movies by Critic Count
top10_critic_count = df[['movie_title', 'tomatometer_count']].dropna()
top10_critic_count = top10_critic_count.sort_values(by='tomatometer_count', ascending=False).head(10)
print("🎬 Top 10 Movies by Critic Count:\n")
print(top10_critic_count)

#4. Top 10 Movies by Audience Count
top10_audience_count = df[['movie_title', 'audience_count']].dropna()
top10_audience_count = top10_audience_count.sort_values(by='audience_count', ascending=False).head(10)
print("\n🎬 Top 10 Movies by Audience Count:\n")
print(top10_audience_count)


# ------------------ 📊 VISUALIZATION SECTION ------------------
color ='darkblue'

numeric_cols = df[['tomatometer_rating', 'audience_rating', 'runtime_in_minutes', 'audience_count', 'tomatometer_count']]
corr = numeric_cols.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.show()

genre_avg = df.groupby('genre')[['audience_rating', 'tomatometer_rating']].mean()
genre_avg['combined_avg'] = (genre_avg['audience_rating'] + genre_avg['tomatometer_rating']) / 2
top10_genres = genre_avg.sort_values(by='combined_avg', ascending=False).head(10)
top10_genres = top10_genres.reset_index()
genre_melted = pd.melt(top10_genres, id_vars='genre', value_vars=['audience_rating', 'tomatometer_rating'],
                       var_name='Rating Type', value_name='Rating')
plt.figure(figsize=(12, 6))
sns.barplot(data=genre_melted, x='Rating', y='genre', hue='Rating Type', palette='Set2')
plt.title("Top 10 Genres by Audience and Critic Rating")
plt.xlabel("Average Rating")
plt.ylabel("Genre")
plt.legend(title="Rating Type")
plt.tight_layout()
plt.show()

yearly = df.groupby('year')[['audience_rating', 'tomatometer_rating']].mean().reset_index()
plt.figure(figsize=(10, 5))
plt.plot(yearly['year'], yearly['audience_rating'], label='Audience Rating', color='skyblue')
plt.plot(yearly['year'], yearly['tomatometer_rating'], label='Critic Rating', color='lightcoral')
plt.title("Average Audience vs Critic Ratings Over the Years")
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

sns.scatterplot(x='tomatometer_rating', y='audience_rating', data=df)
plt.title('Critics vs Audience Ratings')
plt.show()