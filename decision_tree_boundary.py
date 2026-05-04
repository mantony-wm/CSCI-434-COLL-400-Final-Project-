"""
Decision Tree Baseline — 2D Decision Boundary Visualization
=============================================================
Generates a 2D projection of the Decision Tree's decision boundaries
using PCA to reduce the 9-dimensional feature space.

Run:
    python decision_tree_boundary.py
    
Outputs: decision_tree_boundary.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# Load data
df = pd.read_csv('combined_second_flows.csv')
FEATURES = ['Packet_Count','Total_Length','Average_Packet_Interval','Maximum_Packet_Interval',
            'Minimum_Packet_Interval','Average_Packet_Length','Maximum_Packet_Length',
            'Minimum_Packet_Length','Most_Common_Packet_Length']

X = df[FEATURES].values
le = LabelEncoder()
y = le.fit_transform(df['Label'])

# 60/20/20 split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# Train Decision Tree on full 9D space
dt = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=3)
dt.fit(X_train, y_train)

# Project to 2D using PCA
pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

# Train a new Decision Tree on the 2D projection for visualization
dt_2d = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=3)
dt_2d.fit(X_train_2d, y_train)

print("Training complete. Generating visualization...")

# Create meshgrid - ADJUST h TO CONTROL RESOLUTION/MEMORY
# h=0.1 is high quality but memory intensive
# h=0.5 is coarser but more memory efficient
# h=1.0 is very coarse but minimal memory
h = 1.0  

x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

print(f"Meshgrid size: {xx.shape} (reduce h if memory error)")

# Predict on meshgrid
Z = dt_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Styling
BG = "#0F1117"
CARD_BG = "#1A1D27"
TEXT = "#E8EAF0"
SUBTEXT = "#8B90A0"
CLASS_COLORS = ['#4E9BCC', '#2ECC71', '#E74C3C']  # amazon, hulu, youtube
CLASS_NAMES = le.classes_

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#2A2D3A",
    "axes.labelcolor": SUBTEXT,
    "xtick.color": SUBTEXT,
    "ytick.color": SUBTEXT,
    "grid.color": "#2A2D3A",
    "text.color": TEXT,
    "font.family": "monospace",
})

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD_BG)

# Plot decision boundary regions
cmap_light = ListedColormap(['#4E9BCC33', '#2ECC7133', '#E74C3C33'])
ax.contourf(xx, yy, Z, alpha=0.5, cmap=cmap_light)

# Plot decision boundary lines (rectangular for Decision Tree)
ax.contour(xx, yy, Z, colors=['#4E9BCC', '#2ECC71', '#E74C3C'], 
           linewidths=2.5, linestyles='solid', alpha=0.9)

# Plot test points
for i, (color, name) in enumerate(zip(CLASS_COLORS, CLASS_NAMES)):
    mask = y_test == i
    ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1], 
              c=color, label=name.capitalize(), s=50, alpha=0.7, 
              edgecolors='white', linewidths=0.5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
             fontsize=11, color=SUBTEXT, fontfamily='monospace')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
             fontsize=11, color=SUBTEXT, fontfamily='monospace')
ax.set_title('Decision Tree (Baseline) Decision Boundaries — 2D PCA Projection\n'
            'Test Set: 314 Windows | Max Depth: 4', 
            fontsize=14, fontweight='bold', color=TEXT, fontfamily='monospace', pad=20)

ax.grid(True, alpha=0.2)
ax.legend(loc='upper right', framealpha=0.3, facecolor=CARD_BG, 
         edgecolor='#2A2D3A', labelcolor=TEXT, fontsize=10)

ax.spines[['top','right','left','bottom']].set_color('#2A2D3A')

plt.tight_layout()
plt.savefig('decision_tree_boundary.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved: decision_tree_boundary.png")
plt.close()

# Report accuracy
y_pred_2d = dt_2d.predict(X_test_2d)
y_pred_9d = dt.predict(X_test)
acc_2d = (y_pred_2d == y_test).mean() * 100
acc_9d = (y_pred_9d == y_test).mean() * 100

print(f"\nDecision Tree accuracy on 2D projection: {acc_2d:.2f}%")
print(f"Decision Tree accuracy on full 9D space: {acc_9d:.2f}%")
print(f"Information loss from projection: {acc_9d - acc_2d:.2f}pp")
print(f"\nNote: The rectangular decision boundaries are characteristic of")
print(f"tree-based models which split along single features at a time.")
