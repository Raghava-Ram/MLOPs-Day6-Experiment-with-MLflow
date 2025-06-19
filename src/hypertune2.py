from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import pandas as pd

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X.astype('float64'),  # Convert to float64 to avoid schema issues
    y,
    test_size=0.2,
    random_state=42
)

# Creating the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# Defining the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Set MLflow experiment
mlflow.set_experiment('breast-cancer-rf-hp')

with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)

    # Log each child run (param combination)
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])

    # Displaying and logging best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)

    # Save training data
    train_df = X_train.copy()
    train_df["target"] = y_train
    train_df.to_csv("train.csv", index=False)
    mlflow.log_artifact("train.csv", artifact_path="data")

    # Save test data
    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv("test.csv", index=False)
    mlflow.log_artifact("test.csv", artifact_path="data")

    # Log the best model with signature and input example
    input_example = X_train.iloc[:5]
    signature = infer_signature(X_train, grid_search.best_estimator_.predict(X_train))

    mlflow.sklearn.log_model(
        grid_search.best_estimator_,
        name="random_forest",
        input_example=input_example,
        signature=signature
    )

    # Set metadata tags
    mlflow.set_tag("author", "Raghava Ram")

    print(best_params)
    print(best_score)
