import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingClassifier

from imblearn.over_sampling import SMOTE
import shap


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
RANDOM_STATE = 42
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")


# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print(df.head())
    print("\nShape:", df.shape)
    return df


# -------------------------------------------------
# 2. CLEANING
# -------------------------------------------------
def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Basic Cleaning ===")

    drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print("Columns after dropping:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())

    return df


# -------------------------------------------------
# 3. MINIMAL EDA VISUALIZATIONS
# -------------------------------------------------
def exploratory_visualizations(df):
    print("\n=== Minimal EDA Visualizations ===")

    # 1. Attrition Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Attrition", palette="Set2")
    plt.title("Attrition Distribution")
    plt.show()

    #  2. Attrition by Department
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="Department", hue="Attrition", palette="Set1")
    plt.title("Attrition by Department")
    plt.xticks(rotation=45)
    plt.show()

    # 3. Four Key Boxplots
    key_vars = ["Age", "MonthlyIncome", "DistanceFromHome", "TotalWorkingYears"]
    for col in key_vars:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="Attrition", y=col, palette="Set3")
        plt.title(f"{col} vs Attrition")
        plt.show()

    # 4. Correlation Heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(df.select_dtypes(include=np.number).corr(),
                cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.show()


# -------------------------------------------------
# 4. PREPROCESS
# -------------------------------------------------
def preprocess(df: pd.DataFrame):
    print("\n=== Preprocessing ===")

    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    y = df["Attrition"]
    X = df.drop(columns=["Attrition"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    print("Categoricals:", cat_cols)
    print("Numericals:", num_cols)

    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    print("Shape after encoding:", X_encoded.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Scale numericals
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    print("\nAfter SMOTE distribution:")
    print(y_train_bal.value_counts(normalize=True))

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=35)
    X_train_sel = selector.fit_transform(X_train_bal, y_train_bal)
    X_test_sel = selector.transform(X_test_scaled)

    selected_idx = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_idx]

    print("\nSelected features:")
    print(selected_features.tolist())

    return X_train_sel, X_test_sel, y_train_bal, y_test, selected_features.values


# -------------------------------------------------
# 5. TRAIN MODELS
# -------------------------------------------------
def train_models(X_train, y_train):
    print("\n=== Training Models ===")

    models = {}

    # Logistic Regression
    log_reg = LogisticRegression(
        max_iter=1500, class_weight={0: 1, 1: 4}, random_state=RANDOM_STATE
    )
    log_reg.fit(X_train, y_train)
    models["Logistic Regression"] = log_reg

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=20,
        min_samples_split=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    # XGBoost Tuning
    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,         
        tree_method="hist" 
    )

    param_grid = {
        "n_estimators": [300, 500, 700],
        "learning_rate": [0.01, 0.03],
        "max_depth": [4, 6],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        n_jobs=1,   
        verbose=1
    )
    search.fit(X_train, y_train)

    models["XGBoost"] = search.best_estimator_

    # Stacking
    stack = StackingClassifier(
        estimators=[
            ("lr", log_reg),
            ("rf", rf),
            ("xgb", models["XGBoost"]),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
    )
    stack.fit(X_train, y_train)
    models["Stacked Ensemble"] = stack

    return models


# -------------------------------------------------
# 6. EVALUATION
# -------------------------------------------------
def evaluate_model(name, model, X_test, y_test, plot_cm=False):
    print(f"\n=== Evaluation: {name} ===")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if plot_cm:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    return {"model": name, "roc_auc": roc}


def compare_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X_test, y_test, plot_cm=False)
        results.append(res)

    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results_df.sort_values(by="roc_auc", ascending=False))
    return results_df


# -------------------------------------------------
# 7. FEATURE IMPORTANCE + SHAP
# -------------------------------------------------
def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(idx)), importances[idx][::-1])
    plt.yticks(range(len(idx)), feature_names[idx][::-1])
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.show()


def shap_explain_model(model, X_train, feature_names):
    print("\n=== SHAP Summary Plot ===")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train[:200])  # sample
    shap.summary_plot(shap_values, X_train[:200], feature_names=feature_names)
    plt.show()


# -------------------------------------------------
# 8. MAIN
# -------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    df_clean = basic_cleaning(df)

    exploratory_visualizations(df_clean)

    X_train, X_test, y_train, y_test, feature_names = preprocess(df_clean)

    models = train_models(X_train, y_train)
    results_df = compare_models(models, X_test, y_test)

    best_model_name = results_df.sort_values(by="roc_auc", ascending=False).iloc[0]["model"]
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name}")

    # Plot CM ONLY for best model
    evaluate_model(best_model_name, best_model, X_test, y_test, plot_cm=True)

    # Feature importance & SHAP only for tree models
    if best_model_name in ["Random Forest", "XGBoost"]:
        plot_feature_importance(best_model, feature_names)
        shap.initjs()
        shap_explain_model(best_model, X_train, feature_names)


if __name__ == "__main__":
    main()
