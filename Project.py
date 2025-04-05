import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('Rotten Tomatoes Movies.csv')  # Replace with your actual file path

# EDA Process

# ------------------------
# 1. Data Overview
# ------------------------
print(df.head())
print(df.info())
print(df.describe(include='all'))
print("Shape:", df.shape)
print("Null values:\n", df.isnull().sum())

# ------------------------
# 2. Data Cleaning
# ------------------------
df['in_theaters_date'] = pd.to_datetime(df['in_theaters_date'], errors='coerce')
df['on_streaming_date'] = pd.to_datetime(df['on_streaming_date'], errors='coerce')
df['genre'] = df['genre'].astype(str).str.strip()
df['studio_name'] = df['studio_name'].astype(str).str.strip()
df = df.drop_duplicates()

# ------------------------
# 3. Univariate Analysis
# ------------------------
sns.histplot(df['tomatometer_rating'], kde=True)
plt.title('Tomatometer Rating Distribution')
plt.show()

sns.boxplot(x=df['audience_rating'])
plt.title('Audience Rating Boxplot')
plt.show()

sns.countplot(y='genre', data=df, order=df['genre'].value_counts().index[:10])
plt.title('Top 10 Genres')
plt.show()

# ------------------------
# 4. Bivariate Analysis
# ------------------------
sns.scatterplot(x='tomatometer_rating', y='audience_rating', data=df)
plt.title('Critics vs Audience Ratings')
plt.show()

# Genre-wise average audience rating
df.groupby('genre')['audience_rating'].mean().sort_values(ascending=False).head(10).plot(kind='barh')
plt.title('Top Genres by Audience Rating')
plt.xlabel('Avg Audience Rating')
plt.show()

sns.boxplot(x='rating', y='tomatometer_rating', data=df)
plt.title('Rating vs Tomatometer Rating')
plt.xticks(rotation=45)
plt.show()

# ------------------------
# 5. Date-Based Trends
# ------------------------
df['year'] = df['in_theaters_date'].dt.year
df['year'].value_counts().sort_index().plot(kind='line')
plt.title('Movies Released Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.grid(True)
plt.show()

df.groupby('year')[['tomatometer_rating', 'audience_rating']].mean().plot()
plt.title('Average Ratings Over Years')
plt.grid(True)
plt.show()

# ------------------------
# 6. Outlier Detection
# ------------------------
sns.boxplot(x=df['runtime_in_minutes'])
plt.title('Runtime in Minutes')
plt.show()

# ------------------------
# 7. Top Movies by Ratings
# ------------------------
print("Top 10 Critic Rated Movies:")
print(df[['movie_title', 'tomatometer_rating']].sort_values(by='tomatometer_rating', ascending=False).head(10))

print("Top 10 Audience Rated Movies:")
print(df[['movie_title', 'audience_rating']].sort_values(by='audience_rating', ascending=False).head(10))

print("Top 10 Most Reviewed by Critics:")
print(df[['movie_title', 'tomatometer_count']].sort_values(by='tomatometer_count', ascending=False).head(10))

df['in_theaters_date'] = pd.to_datetime(df['in_theaters_date'], errors='coerce')
df['on_streaming_date'] = pd.to_datetime(df['on_streaming_date'], errors='coerce')
df['year'] = df['in_theaters_date'].dt.year


# Visulaization

# 8. Trend of average ratings over the years
ratings_over_years = df.groupby('year')[['tomatometer_rating', 'audience_rating']].mean()
ratings_over_years.plot(title='Average Ratings Over Years', figsize=(10, 6))
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()

# 9. Heatmap of average ratings by genre
genre_rating = df.groupby('genre')[['tomatometer_rating', 'audience_rating']].mean().fillna(0)
plt.figure(figsize=(10, 8))
sns.heatmap(genre_rating, annot=True, fmt=".1f", cmap="coolwarm")
plt.title('Average Ratings by Genre')
plt.show()

# 10. Top 10 studios by movie count
top_studios = df['studio_name'].value_counts().head(10)
top_studios.plot(kind='bar', figsize=(10, 6), title='Top 10 Studios by Number of Movies')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.show()

# 11. Scatter plot with regression line: Critics vs Audience Ratings
plt.figure(figsize=(8, 6))
sns.regplot(x='tomatometer_rating', y='audience_rating', data=df, scatter_kws={'alpha':0.5})
plt.title('Critics vs Audience Ratings')
plt.show()

# 12. Correlation heatmap for numerical values
numeric_cols = df[['tomatometer_rating', 'audience_rating', 'runtime_in_minutes', 'audience_count', 'tomatometer_count']]
corr = numeric_cols.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.show()

# 13. Scatter plot of runtime vs tomatometer rating
plt.figure(figsize=(8, 6))
sns.scatterplot(x='runtime_in_minutes', y='tomatometer_rating', data=df)
plt.title('Movie Runtime vs Tomatometer Rating')
plt.show()
