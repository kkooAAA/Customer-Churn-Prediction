import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("customer_churn.csv")

X = df.drop(columns=["churn", "customer_id"])
y = (df["churn"] == "Yes").astype(int)

num_cols = X[["senior_citizen", "tenure", "monthly_charges", "total_charges"]].columns
cat_cols = X.drop(columns=["senior_citizen", "tenure", "monthly_charges", "total_charges"]).columns

num_pipelines = Pipeline([
    ("imp", SimpleImputer(strategy="mean")),
    ("sc", StandardScaler())
])

cat_pipelines = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent"))
    ,("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])

prep = ColumnTransformer([
    ("num", num_pipelines, num_cols),
    ("cat", cat_pipelines, cat_cols)
], verbose_feature_names_out=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "LR": LogisticRegression(max_iter=1000),
    "RF": RandomForestClassifier(random_state=42)
}

param_grids = {
    "LR": {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__fit_intercept": [True, False],
        "model__class_weight": ["balanced", None],
    },
    "RF": {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_leaf": [1, 3, 5],
        "model__class_weight": ["balanced", None],
    },
}

best_name = None
best_model = None
max_score = -1

for name, model in models.items():
    pipe = Pipeline([
        ("prep", prep),
        ('model', model)
    ])
    
    scores = cross_val_score(
        pipe,
        X_train, y_train,
        cv=5,
        scoring="f1"
    )
    
    model_score = scores.mean()
    print(f"{name} CV F1: {model_score:.4f}")
    if model_score > max_score:
        best_name = name
        best_model = model
        max_score = model_score

print(f"Best model: {best_name}")

search = GridSearchCV(
    Pipeline([("prep", prep), ("model", best_model)]),
    param_grid=param_grids[best_name],
    cv=5, 
    scoring="f1", 
    n_jobs=-1, 
    verbose=1
)
search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print(f"Best CV F1: {search.best_score_:.4f}")

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))