from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "credit-risk-rf-best"

# Get the latest version of the model (can filter for "Staging" or "None" stage)
latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version

current_stage = client.get_model_version(model_name, latest_version).current_stage

if current_stage != "Production":
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )
    print(f"✅ Model '{model_name}' version {latest_version} promoted to Production")
else:
    print(f"⚠️ Model '{model_name}' version {latest_version} is already in Production.")
    
