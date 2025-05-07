#importing panda,plt and sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load dataset into a pandas DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]  # Add target labels
#data inspection
print("First 5 rows:")
display(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())
#statistical part
print("Summary statistics:")
display(df.describe())
#grouping by species
species_mean = df.groupby('species').mean()
print("\nMean of numerical features by species:")
display(species_mean)
#   notable observation
#Setosa has the smallest petal dimensions.
#Virginica has the largest sepal and petal measurements.
#task 3-data visualization
plt.figure(figsize=(8, 4))
df_sample = df.sample(10).sort_values('sepal length (cm)')
plt.plot(df_sample['sepal length (cm)'], marker='o')
plt.title("Sample Trend in Sepal Length")
plt.xlabel("Observation Index")
plt.ylabel("Sepal Length (cm)")
plt.grid(True)
plt.show()
#bar chart
plt.figure(figsize=(6, 4))
species_mean['sepal length (cm)'].plot(kind='bar', color=['blue', 'green', 'red'])
plt.title("Average Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Mean Sepal Length (cm)")
plt.xticks(rotation=0)
plt.show()
#histogram-distribution of petal width
plt.figure(figsize=(6, 4))
df['petal width (cm)'].hist(bins=15, color='purple', edgecolor='black')
plt.title("Distribution of Petal Width")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Frequency")
plt.show()
#Scatter Plot (Sepal Length vs. Petal Length)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='viridis')
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.show()
'''notable findings: 
Setosa has distinctly smaller petals compared to Versicolor and Virginica.

Virginica has the largest sepals and petals on average.

Petal width is bimodally distributed (likely separating Setosa from the other species).

Sepal length and petal length show a strong positive correlation.'''