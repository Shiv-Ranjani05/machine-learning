from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

X,y = make_moons(n_samples=300, noise=0.3, random_state=42)

df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df['Target'] = y
# Visualize the 2D data
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Feature 1", y="Feature 2", hue="Target", palette="Set1")
plt.title("2D Classification Data (make_moons)")
plt.grid(True)
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.3, random_state=42, stratify=y
)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Train a k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# Predict and evaluate
y_pred = knn.predict(X_test)
print(f"Test Accuracy (k=5): {accuracy_score(y_test, y_pred):.2f}")


