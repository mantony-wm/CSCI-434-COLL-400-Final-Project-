"""
Network Traffic Classification — Model Training & Evaluation
=============================================================
Trains 6 models (Logistic Regression, Decision Tree, SVM, Random Forest, 
Gradient Boosting, Ensemble) on 1-second window traffic features.

Outputs:
- model_results.pkl: Contains all results, confusion matrices, per-class metrics
- Prints test accuracy and F1-macro for each model

Run:
    python train_models.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                              classification_report, confusion_matrix)
import pickle
import warnings
warnings.filterwarnings('ignore')

# ========== Load Data ==========
df = pd.read_csv('combined_second_flows.csv')
print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['Label'].value_counts()}\n")

FEATURES = ['Packet_Count','Total_Length','Average_Packet_Interval','Maximum_Packet_Interval',
            'Minimum_Packet_Interval','Average_Packet_Length','Maximum_Packet_Length',
            'Minimum_Packet_Length','Most_Common_Packet_Length']

X = df[FEATURES].values
le = LabelEncoder()
y = le.fit_transform(df['Label'])
print(f"Classes: {le.classes_}\n")

# ========== 60/20/20 Train/Val/Test Split ==========
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Split sizes:")
print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)\n")

# ========== Feature Scaling (for SVM and Logistic Regression) ==========
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ========== Evaluation Function ==========
def evaluate(model, X_tr, X_v, X_te, y_tr, y_v, y_te, model_name):
    """Train model and compute metrics on train/val/test sets"""
    print(f"Training {model_name}...")
    model.fit(X_tr, y_tr)
    
    results = {}
    for split_name, Xs, ys in [('train', X_tr, y_tr), ('val', X_v, y_v), ('test', X_te, y_te)]:
        pred = model.predict(Xs)
        results[split_name] = {
            'accuracy':    round(accuracy_score(ys, pred) * 100, 2),
            'f1_macro':    round(f1_score(ys, pred, average='macro') * 100, 2),
            'f1_weighted': round(f1_score(ys, pred, average='weighted') * 100, 2),
            'precision':   round(precision_score(ys, pred, average='macro') * 100, 2),
            'recall':      round(recall_score(ys, pred, average='macro') * 100, 2),
        }
    
    # Per-class metrics on test set
    test_pred = model.predict(X_te)
    report = classification_report(y_te, test_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_te, test_pred)
    
    print(f"  Test Acc: {results['test']['accuracy']:.2f}%  |  F1-Macro: {results['test']['f1_macro']:.2f}%")
    
    return results, report, cm, model

# ========== Train All Models ==========
all_results = {}

# 1. Logistic Regression
res, rep, cm, mdl = evaluate(
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_s, X_val_s, X_test_s, y_train, y_val, y_test,
    "Logistic Regression"
)
all_results['Logistic Regression'] = (res, rep, cm)

# 2. Decision Tree (Baseline)
dt = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=3)
res, rep, cm, mdl = evaluate(
    dt, X_train, X_val, X_test, y_train, y_val, y_test,
    "Decision Tree"
)
all_results['Decision Tree'] = (res, rep, cm)

# Export decision tree visualization
export_graphviz(dt, out_file='decision_tree.dot',
                feature_names=[f.replace('_', ' ') for f in FEATURES],
                class_names=le.classes_, filled=True, rounded=True, precision=2)
print("  Decision tree exported to decision_tree.dot (use graphviz to render)")

# 3. SVM
res, rep, cm, mdl = evaluate(
    SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    X_train_s, X_val_s, X_test_s, y_train, y_val, y_test,
    "SVM"
)
all_results['SVM'] = (res, rep, cm)

# 4. Random Forest
res, rep, cm, mdl = evaluate(
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    X_train, X_val, X_test, y_train, y_val, y_test,
    "Random Forest"
)
all_results['Random Forest'] = (res, rep, cm)

# 5. Gradient Boosting
res, rep, cm, mdl = evaluate(
    GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
    X_train, X_val, X_test, y_train, y_val, y_test,
    "Gradient Boosting"
)
all_results['Gradient Boosting'] = (res, rep, cm)

# 6. Ensemble (Soft Voting: SVM + RF + GB)
print("Training Ensemble (SVM+RF+GB)...")
svm_ens = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
rf_ens  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
gb_ens  = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
ensemble = VotingClassifier(
    estimators=[('svm', svm_ens), ('rf', rf_ens), ('gb', gb_ens)],
    voting='soft'
)
res, rep, cm, mdl = evaluate(
    ensemble, X_train_s, X_val_s, X_test_s, y_train, y_val, y_test,
    "Ensemble (SVM+RF+GB)"
)
all_results['Ensemble (SVM+RF+GB)'] = (res, rep, cm)

# ========== Save Results ==========
output_data = {
    'all_results': all_results,
    'classes': le.classes_.tolist(),
    'splits': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
    'features': FEATURES,
}

with open('model_results.pkl', 'wb') as f:
    pickle.dump(output_data, f)

print("\n" + "="*60)
print("SUMMARY — Test Set Performance")
print("="*60)
for model_name in ['Logistic Regression', 'Decision Tree', 'SVM', 
                     'Random Forest', 'Gradient Boosting', 'Ensemble (SVM+RF+GB)']:
    res = all_results[model_name][0]
    print(f"{model_name:25s}  Acc: {res['test']['accuracy']:6.2f}%  F1: {res['test']['f1_macro']:6.2f}%")

print("\nResults saved to model_results.pkl")
print("Decision tree saved to decision_tree.dot")
print("\nTo generate visualizations, run:")
print("  dot -Tpng decision_tree.dot -o decision_tree.png")
