from azureml.core import Experiment
from azureml.core.run import Run

def get_last_run(workspace, experiment_name):
    experiment = Experiment(workspace, experiment_name)

    for run in experiment.get_runs():
        if(run.get_status()=='Completed'):
            return run

experiment_name = 'mlops-churn'

run = Run.get_context()
workspace = run.experiment.workspace

last_run = get_last_run(workspace, experiment_name)

last_run.register_model(model_name='churn-model', 
                        model_path='outputs/model.pickle',
                        description='Churn Model',
                        tags={'Model': 'XGBoost', 'type': 'Classification'},
                          )