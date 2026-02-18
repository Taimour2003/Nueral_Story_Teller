# ============================================
# BASIC MODEL TRAINING - STEP BY STEP GUIDE
# ============================================

# 1. IMPORT ALL NECESSARY LIBRARIES
print("=" * 50)
print("Step 1: Importing Libraries")
print("=" * 50)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import warnings
import torch.nn as nn
import torch

# warnings.filterwarnings("ignore")

# print("✓ All libraries imported successfully!\n")

# # 2. LOAD DATASET
# print("=" * 50)
# print("Step 2: Loading Dataset")
# print("=" * 50)

# # Using Iris dataset (classic ML dataset)
# iris = load_iris()
# X = iris.data  # Features
# y = iris.target  # Target labels

# print(f"Dataset loaded: Iris Dataset")
# print(f"Number of samples: {X.shape[0]}")
# print(f"Number of features: {X.shape[1]}")
# print(f"Feature names: {iris.feature_names}")
# print(f"Target classes: {iris.target_names}")
# print(f"\nFirst 5 samples of features:\n{X[:5]}")
# print(f"\nFirst 5 labels: {y[:5]}\n")

# # 3. DATA EXPLORATION
# print("=" * 50)
# print("Step 3: Data Exploration")
# print("=" * 50)

# df = pd.DataFrame(X, columns=iris.feature_names)
# df['target'] = y

# print("Dataset Statistics:")
# print(df.describe())
# print(f"\nClass distribution:")
# print(df['target'].value_counts())
# print()

# # 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# print("=" * 50)
# print("Step 4: Splitting Data")
# print("=" * 50)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y
# )

# print(f"Training set size: {X_train.shape[0]} samples")
# print(f"Testing set size: {X_test.shape[0]} samples")
# print(f"Training set shape: {X_train.shape}")
# print(f"Testing set shape: {X_test.shape}\n")

# # 5. FEATURE SCALING
# print("=" * 50)
# print("Step 5: Feature Scaling")
# print("=" * 50)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# print("Before scaling (first sample):", X_train[0])
# print("After scaling (first sample):", X_train_scaled[0])
# print("✓ Features scaled (normalized)\n")

# # 6. CREATE AND TRAIN THE MODEL
# print("=" * 50)
# print("Step 6: Training the Model")
# print("=" * 50)

# model = LogisticRegression(max_iter=200, random_state=42)
# print(f"Model: {model.__class__.__name__}")
# print("Training started...")

# model.fit(X_train_scaled, y_train)

# print("✓ Model training completed!\n")

# # 7. MAKE PREDICTIONS
# print("=" * 50)
# print("Step 7: Making Predictions")
# print("=" * 50)

# y_train_pred = model.predict(X_train_scaled)
# y_test_pred = model.predict(X_test_scaled)

# print("Sample predictions on test set:")
# print(f"Actual: {y_test[:10]}")
# print(f"Predicted: {y_test_pred[:10]}\n")

# # 8. EVALUATE THE MODEL
# print("=" * 50)
# print("Step 8: Model Evaluation")
# print("=" * 50)

# train_accuracy = accuracy_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_test_pred)

# print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
# print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
# print()

# print("Detailed Classification Report:")
# print(classification_report(y_test, y_test_pred, target_names=iris.target_names))

# print("Confusion Matrix:")
# cm = confusion_matrix(y_test, y_test_pred)
# print(cm)
# print()

# # 9. VISUALIZE RESULTS (Optional)
# print("=" * 50)
# print("Step 9: Visualization")
# print("=" * 50)

# plt.figure(figsize=(8, 6))
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap='viridis',
#             edgecolors='k', s=100, alpha=0.7)
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[1])
# plt.title('Model Predictions (First 2 Features)')
# plt.colorbar(label='Predicted Class')
# print("✓ Visualization created (close the plot window to continue)\n")
# plt.show()

# # 10. SUMMARY
# print("=" * 50)
# print("TRAINING COMPLETE - SUMMARY")
# print("=" * 50)
# print(f"✓ Dataset: Iris (150 samples, 4 features, 3 classes)")
# print(f"✓ Model: Logistic Regression")
# print(f"✓ Training Accuracy: {train_accuracy * 100:.2f}%")
# print(f"✓ Testing Accuracy: {test_accuracy * 100:.2f}%")
# print(f"✓ Model is ready for deployment!")
# print("=" * 50)


hidden_state = torch.randn(4)  # Example hidden state vector
# print("Hidden state shape:", hidden_state.shape)
# print("Hidden state:", hidden_state)
h1 = hidden_state.unsqueeze(0)
# print("Unsqueezed hidden state shape:", h1.shape)
# print("Unsqueezed hidden state:", h1)
h = h1.repeat(2, 1, 1)
print("Repeated hidden state shape:", h.shape)
print("Repeated hidden state:", h)
c = torch.zeros_like(h)
print("Cell state shape:", c.shape)
print("Cell state:", c)
