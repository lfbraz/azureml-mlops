from azureml.core import Environment
from azureml.core.run import Run
from azureml.core.model import InferenceConfig, Model
from azureml.core.compute import AksCompute

run = Run.get_context()
workspace = run.experiment.workspace

xgboost_env = Environment.get(workspace, 'XGBoost-Env')

inference_config = InferenceConfig(entry_script="entry.py",
                                   environment=xgboost_env)

cluster_name = 'aks-e2e-1'
aks_target = AksCompute(workspace, cluster_name)

model = workspace.models['churn-model']

service = Model.deploy(workspace=workspace,
                       name = 'api-churn-model-prod',
                       models = [model],
                       inference_config = inference_config,
                       deployment_target = aks_target,
                       overwrite=True)
service.wait_for_deployment(show_output = True)