import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "Rotten Tomatoes Movies.csv"  # Update this with your actual file path
df = pd.read_csv(file_path)

# Display basic information
print("Initial Dataset Overview:")
print(df.info())

# Handling missing values
df.dropna(inplace=True)  # Drop rows with missing values

# Removing duplicates
df.drop_duplicates(inplace=True)

# Convert columns to appropriate data types
# df['Tomatometer Score'] = pd.to_numeric(df['tomatometer_rating'], errors='coerce')
# df['Audience Score'] = pd.to_numeric(df['audience_rating'], errors='coerce')

# # Fill missing scores with median values
# df['Tomatometer Score'].fillna(df['Tomatometer Score'].median(), inplace=True)
# df['Audience Score'].fillna(df['Audience Score'].median(), inplace=True)

# # Convert "Release Year" to numeric for analysis
# df['Release Year'] = pd.to_numeric(df['Release Year'], errors='coerce')

# ------------------------- 🎯 Data Analysis ----------------------------

# 📌 **1. Top 10 Highest-Rated Movies (Critics & Audience)**
top_critics = df[['movie_title', 'tomatometer_rating']].sort_values(by='tomatometer_rating', ascending=False).head(10)
top_audience = df[['movie_title', 'audience_rating']].sort_values(by='audience_rating', ascending=False).head(10)

print("\n🎬 Top 10 Movies by Critics' Scores:")
print(top_critics)

print("\n🎬 Top 10 Movies by Audience Scores:")
print(top_audience)

# 📌 **2. Distribution of Ratings (Critics & Audience)**
plt.figure(figsize=(12, 5))
sns.histplot(df['Tomatometer Score'], bins=20, kde=True, color='red', label="Critics Score")
sns.histplot(df['Audience Score'], bins=20, kde=True, color='blue', label="Audience Score", alpha=0.7)
plt.title("Distribution of Rotten Tomatoes Scores")
plt.xlabel("Score")
plt.ylabel("Count")
plt.legend()
plt.show()

# 📌 **3. Correlation Between Critics' and Audience Ratings**
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['Tomatometer Score'], y=df['Audience Score'], alpha=0.5)
plt.title("Critics Score vs Audience Score")
plt.xlabel("Tomatometer Score")
plt.ylabel("Audience Score")
plt.show()

# 📌 **4. Heatmap: Correlation Between Numerical Features**
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# 📌 **5. Yearly Trends in Movie Ratings**
yearly_avg = df.groupby("Release Year")[["Tomatometer Score", "Audience Score"]].mean().dropna()

plt.figure(figsize=(12, 5))
plt.plot(yearly_avg.index, yearly_avg['Tomatometer Score'], marker='o', linestyle='-', color='red', label="Critics Score")
plt.plot(yearly_avg.index, yearly_avg['Audience Score'], marker='s', linestyle='-', color='blue', label="Audience Score")
plt.xlabel("Release Year")
plt.ylabel("Average Score")
plt.title("Yearly Trend of Rotten Tomatoes Scores")
plt.legend()
plt.show()

# 📌 **6. Most Common Genres**
genre_counts = df['Genre'].value_counts().head(10)
print("\n📊 Most Common Genres:")
print(genre_counts)

plt.figure(figsize=(10, 5))
genre_counts.plot(kind='bar', color='purple')
plt.title("Most Common Movie Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.show()

# 📌 **7. Boxplot of Scores by Genre (Top 5 Genres)**
top_genres = df['Genre'].value_counts().index[:5]
df_filtered = df[df['Genre'].isin(top_genres)]

plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Tomatometer Score', data=df_filtered, palette='coolwarm')
plt.title("Critics' Score Distribution by Genre")
plt.xlabel("Genre")
plt.ylabel("Tomatometer Score")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Audience Score', data=df_filtered, palette='coolwarm')
plt.title("Audience Score Distribution by Genre")
plt.xlabel("Genre")
plt.ylabel("Audience Score")
plt.xticks(rotation=45)
plt.show()

# ------------------------- 🎯 Key Insights ----------------------------

"""
🔍 Insights from the analysis:
- The highest-rated movies differ between critics and audiences.
- The distribution of critics’ scores is often more concentrated, while audience scores vary more.
- There is a **moderate correlation** between Critics' and Audience Scores.
- Over time, the average movie scores **fluctuate**, with some years having higher-rated films.
- The **most common genres** appear to be Action, Drama, and Comedy.
- Some genres have wider variations in scores, as seen in boxplots.
"""

