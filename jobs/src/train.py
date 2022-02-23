from azureml.core import Workspace, Dataset
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import numpy as np
import seaborn as sns

from azureml.core.run import Run

import argparse
import os
import pickle

run = Run.get_context()


def get_dataset(workspace, name):
    dataset = Dataset.get_by_name(workspace, name=name)
    return dataset.to_pandas_dataframe()

def preprocessing(dataset):
    numeric_columns = []
    
    for col in dataset.columns:
        if(dataset[col].dtypes!='object'):
            numeric_columns.append(col)

    dataset = dataset.dropna()
    return dataset, numeric_columns
  
def split_dataset(dataset, seed, test_size=0.33):
    train_dataset, test_dataset = train_test_split(dataset, random_state=seed, test_size=test_size)
    return train_dataset, test_dataset

def get_X_y(train, test, target_column, numeric_columns, drop_columns):
  X_train = train[numeric_columns].drop(drop_columns, axis=1)
  X_test = test[numeric_columns].drop(drop_columns, axis=1)

  y_train = train[target_column]
  y_test = test[target_column]
  return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test, params):
    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)

    model = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                       evals=[(test, "test")], early_stopping_rounds=50, verbose_eval=False)
    return model

def generate_image(run, dataset):
    plot = sns.distplot(dataset.IndiceSatisfacao, kde=False)
    fig = plot.get_figure()
    fig.savefig('outputs/dist-plot-satisfaction-index.jpg', bbox_inches='tight')

    run.log_image(name='dist-plot-satisfaction-index', 
                  path='outputs/dist-plot-satisfaction-index.jpg',                   
                  description='Plot a histogram of the dependent variable IndiceSatisfacao')

def persist_model(model):
    os.makedirs('outputs', exist_ok=True)
    pickle.dump(model, open("outputs/model.pickle", "wb"))

def get_parameters(max_depth, learning_rate, reg_alpha, reg_lambda, min_child_weight, seed):
    params = {'early_stopping_rounds': 50, 
              'learning_rate': learning_rate, 
              'max_depth': max_depth, 
              'maximize': False, 
              'min_child_weight': min_child_weight, 
              'num_boost_round': 1000, 
              'reg_alpha': reg_alpha, 
              'reg_lambda': reg_lambda, 
              'objective': 'binary:hinge',
              'seed': seed}
    return params

def main():
    workspace = run.experiment.workspace
    seed = 2022
    target = 'Churn'
    drop_columns = [target, 'CodigoCliente']
    dataset_name = 'Clientes'

    parser = argparse.ArgumentParser()

    parser.add_argument('--max_depth', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--reg_alpha', type=float, default=1.0)
    parser.add_argument('--reg_lambda', type=float, default=1.0)
    parser.add_argument('--min_child_weight', type=float, default=1.0)

    args = parser.parse_args()

    run.log('max_depth', np.float(args.max_depth))
    run.log('learning_rate', np.int(args.learning_rate))
    run.log('reg_alpha', np.float(args.reg_alpha))
    run.log('reg_lambda', np.float(args.reg_lambda))
    run.log('min_child_weight', np.float(args.min_child_weight))
    
    params = get_parameters(args.max_depth, args.learning_rate, args.reg_alpha, args.reg_lambda, args.min_child_weight, seed)

    # Get the Train Dataset
    dataset = get_dataset(workspace, dataset_name)

    # Preprocessing Features
    dataset, numeric_columns = preprocessing(dataset)

    # Persist Distribution image
    generate_image(run, dataset)

    # Split train and test
    train_dataset, test_dataset = split_dataset(dataset, seed)

    # Get X, y
    X_train, X_test, y_train, y_test = get_X_y(train_dataset, test_dataset, target, numeric_columns, drop_columns)

    # Train Model
    model = train_model(X_train, y_train, X_test, y_test, params)

    # Predict Test dataset
    test = xgb.DMatrix(data=X_test, label=y_test)
    predictions_test = model.predict(test)
    auc_score = roc_auc_score(y_test, predictions_test)
    run.log('auc', round(auc_score, 4))

    # Persist the model
    persist_model(model)
    print('Model-V2')

if __name__ == '__main__':
    main()

