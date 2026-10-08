import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATASET_PATH = "pca_face_dataset"
SIZE, MIN_IMAGES = 64, 5          # images are resized to 64x64
os.makedirs("results", exist_ok=True)


def save_and_show(name):
    plt.tight_layout()
    plt.savefig("results/" + name + ".png", dpi=150)
    plt.show()


# ---------------- STEP 1: Load the face dataset ----------------
if os.path.isdir(DATASET_PATH):
    X, names = [], []
    for person in sorted(os.listdir(DATASET_PATH)):
        folder = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(files) < MIN_IMAGES:
            continue
        for f in files:
            img = Image.open(os.path.join(folder, f)).convert("L").resize((SIZE, SIZE))
            X.append(np.array(img).flatten() / 255.0)
            names.append(person.replace("_", " "))
    X = np.array(X)
    people, y = np.unique(names, return_inverse=True)
    print("Dataset: local folder", DATASET_PATH)
else:
    data = fetch_olivetti_faces()          # 40 people x 10 images, 64x64
    X, y = data.data, data.target
    people = np.array(["Person " + str(i) for i in range(40)])
    print("Dataset: Olivetti Faces (AT&T Laboratories Cambridge)")
print("Images:", len(X), "| People:", len(people), "| Pixels per image:", X.shape[1])

# ---------------- STEP 2: Train / test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
print("Training images:", len(X_train), "| Testing images:", len(X_test))

# ---------------- STEP 3: Eigenfaces ----------------
pca = PCA(n_components=min(20, len(X_train))).fit(X_train)

plt.figure(figsize=(10, 5))
plt.subplot(2, 5, 1)
plt.imshow(pca.mean_.reshape(SIZE, SIZE), cmap="gray")
plt.title("Mean face", fontsize=8)
plt.axis("off")
for i in range(min(9, len(pca.components_))):
    plt.subplot(2, 5, i + 2)
    plt.imshow(pca.components_[i].reshape(SIZE, SIZE), cmap="gray")
    plt.title("Eigenface " + str(i + 1), fontsize=8)
    plt.axis("off")
plt.suptitle("Mean Face and Eigenfaces")
save_and_show("eigenfaces")

# ---------------- STEP 4: 2D PCA projection plot ----------------
points = PCA(n_components=2).fit_transform(X_train)
plt.figure(figsize=(8, 6))
for p in np.unique(y_train):
    plt.scatter(points[y_train == p, 0], points[y_train == p, 1], label=people[p], alpha=0.8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Projection of Training Faces")
if len(people) <= 15:
    plt.legend(fontsize=7)
plt.grid(alpha=0.3)
save_and_show("pca_2d_projection")

# ---------------- STEP 5: Accuracy for different numbers of components ----------------
max_n = min(len(X_train), X.shape[1])
results = {}
for n in [n for n in [2, 5, 10, 15, 20, 30, 50] if n <= max_n]:
    p = PCA(n_components=n).fit(X_train)
    model = KNeighborsClassifier(n_neighbors=1).fit(p.transform(X_train), y_train)
    pred = model.predict(p.transform(X_test))
    acc = (pred == y_test).mean() * 100
    var = p.explained_variance_ratio_.sum() * 100
    results[n] = (acc, var, pred)
    print("Components:", n, "| Accuracy:", round(acc, 2), "% | Variance kept:", round(var, 2), "%")

with open("results/accuracy_vs_components.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["components", "accuracy_percent", "variance_retained_percent"])
    for n, (acc, var, _) in results.items():
        w.writerow([n, round(acc, 2), round(var, 2)])

plt.figure(figsize=(7, 5))
plt.plot(list(results), [r[0] for r in results.values()], marker="o")
plt.xlabel("Number of principal components")
plt.ylabel("Recognition accuracy (%)")
plt.title("Accuracy vs Number of Principal Components")
plt.grid(alpha=0.3)
save_and_show("accuracy_vs_components")

# ---------------- STEP 6: Actual vs predicted labels (best model) ----------------
best = max(results, key=lambda n: results[n][0])
pred = results[best][2]
print("\nBest number of components:", best, "| Accuracy:", round(results[best][0], 2), "%")
print("\nActual vs Predicted:")
with open("results/predictions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test_image", "actual", "predicted", "correct"])
    for i, (a, p) in enumerate(zip(y_test, pred)):
        ok = "yes" if a == p else "no"
        print(i + 1, "| Actual:", people[a], "| Predicted:", people[p], "|", ok)
        w.writerow([i + 1, people[a], people[p], ok])

plt.figure(figsize=(10, 5))
for i in range(min(10, len(X_test))):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(SIZE, SIZE), cmap="gray")
    plt.title("Actual: " + people[y_test[i]] + "\nPred: " + people[pred[i]],
              fontsize=7, color="green" if y_test[i] == pred[i] else "red")
    plt.axis("off")
plt.suptitle("Actual vs Predicted (green = correct, red = wrong)")
save_and_show("predictions")

print("\nDone. Plots and CSV files are saved in the 'results' folder.")