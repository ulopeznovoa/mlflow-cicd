import os
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Obtener variables de entorno
tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
commit_sha = os.environ.get("GITHUB_SHA", "local")
repository = os.environ.get("GITHUB_REPOSITORY", "local")
run_id = os.environ.get("GITHUB_RUN_ID", "local")
print(f"MLflow tracking URI: {tracking_uri}")

# Configurar MLflow
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("demo-ci-cd")

# Preparar los datos
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parametros del modelo
params = {"max_iter": 100, "C": 0.01}

# Crear explícitamente una ejecución
mlflow.start_run(run_name=f"regression-C-{params['C']}")

model = LogisticRegression(**params)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# Registro manual
mlflow.log_params(params)
mlflow.log_metric("accuracy", accuracy)
mlflow.set_tags({
        "git.commit": commit_sha,
        "github.repository": repository,
        "github.run_id": run_id
    })

# Comprobacion de umbral
if accuracy < 0.80:
    mlflow.set_tag("validation.status", "rejected")
    raise RuntimeError(
        f"Precision insuficiente: {accuracy:.3f}"
    )

# Guardar modelo
mlflow.set_tag("validation.status", "accepted")
mlflow.sklearn.log_model(
    sk_model=model, name="model", serialization_format="skops"
)

# Finalizar ejecución
print("Accuracy:", accuracy)
mlflow.end_run()
