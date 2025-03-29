import os 
import sys 
import json
import shutil
import random

import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, make_scorer, f1_score
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATASET_X = '../data/dataset_x.csv'
DATASET_Y = '../data/dataset_y.csv'
DATASET_CONFIG = '../data/dataset_config.json'
RANDOM_STATE = 1234


class XGBoostMulticlassClassifier:
    def __init__(self, labels, training_dir, use_instance_weights=False, manual_weights=None, standardize=False, random_state=RANDOM_STATE):
        self.labels = labels

        if os.path.exists(training_dir):
            print('Existring training directory: {}..'.format(training_dir))
        else:
            os.makedirs(training_dir)
        self.training_dir = training_dir

        self.use_instance_weights = use_instance_weights
        self.manual_weights = manual_weights

        self.standardize = standardize

        self.random_state = random_state
        self.model = None  # Will hold the trained model

    def train(self, X, y, scoring, use_grid_search=True):
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Compute instance weights if present
        sample_weights = None
        if self.use_instance_weights:
            if self.manual_weights:
                sample_weights = np.array([self.manual_weights[label] for label in y_train.iloc[:, 0]])
            else:
                class_weights = self._compute_class_weights(y_train)
                sample_weights = np.array([class_weights[int(label)] for label in y_train.iloc[:, 0]])

        # Perform hyperparameter tuning if enabled
        best_params = self.tune_hyperparameters(X_train, y_train, sample_weights, scoring) if use_grid_search else {}

        # Train final model
        self.model = self._define_model(best_params)
        if self.standardize:
            self.model.fit(X_train, y_train, model__sample_weight=sample_weights)
        else:
            self.model.fit(X_train, y_train, sample_weight=sample_weights)

        # Evaluate model
        self.evaluate(X_train, y_train, 'training')
        self.evaluate(X_test, y_test, 'test')

    def _define_model(self, params=None):
        default_params = {
            'objective': 'multi:softmax',
            'num_class': len(self.labels),
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'random_state': self.random_state
        }
        if params:
            default_params.update(params)

        # Define the model
        model = xgb.XGBClassifier(**default_params)

        # If self.standardize is True, add StandardScaler to the pipeline
        if self.standardize:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            return pipeline
        else:
            return model

    def _compute_class_weights(self, y_train):
        class_counts = Counter(y_train.iloc[:, 0])
        total_samples = y_train.shape[0]
        class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
        
        print(f"Computed class weights: {class_weights}")
        return class_weights

    def tune_hyperparameters(self, X_train, y_train, sample_weights, scoring):
        if self.standardize:
            param_grid = {
                'model__max_depth': [3, 5, 7],
                'model__n_estimators': [50, 100, 200],
            }
        else:
            param_grid = {
                'max_depth': [3, 5, 7],
                'n_estimators': [50, 100, 200],
            }

        base_model = self._define_model()
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=2)

        if sample_weights is not None:
            if self.standardize:
                grid_search.fit(X_train, y_train, model__sample_weight=sample_weights)
            else:
                grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            grid_search.fit(X_train, y_train)
        
        return grid_search.best_params_

    def evaluate(self, X, y, dataset_name):
        if self.model is None:
            raise ValueError("Model is not trained. Call model.train() first.")

        y_pred = self.model.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        report = classification_report(y, y_pred, target_names=[str(label) for label in self.labels], zero_division=0)
        conf_matrix = confusion_matrix(y, y_pred)

        # Save logs and reports with dataset name
        log_data = {
            "dataset": dataset_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix.tolist()
        }

        log_path = os.path.join(self.training_dir, f"{dataset_name}_training_log.json")
        report_path = os.path.join(self.training_dir, f"{dataset_name}_classification_report.txt")
        matrix_path = os.path.join(self.training_dir, f"{dataset_name}_confusion_matrix.png")

        with open(log_path, "w") as log_file:
            json.dump(log_data, log_file, indent=4)

        with open(report_path, "w") as report_file:
            report_file.write(report)

        # Print logs
        print(f'[{dataset_name.upper()}] Accuracy: {accuracy:.4f}')
        print(f'[{dataset_name.upper()}] Precision: {precision:.4f}')
        print(f'[{dataset_name.upper()}] Recall: {recall:.4f}')
        print(f'[{dataset_name.upper()}] F1 Score: {f1:.4f}')
        print(f'[{dataset_name.upper()}] Classification Report:')
        print(report)

        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix ({dataset_name})')
        plt.tight_layout()
        plt.savefig(matrix_path)


if __name__ == "__main__":

    # load data
    print('Loading the dataset configuration..')
    with open(DATASET_CONFIG, 'r') as fp:
        config = json.load(fp)
        labels = config['labels']
        ids = config['ids']

        print("Labels: {}".format(labels))
        print("Number of methods (ids): {}".format(len(ids)))

    print('Loading feature set..')
    X = pd.read_csv(DATASET_X)
    print(X.head())
    print(X.shape)

    print('Loading labels..')
    y = pd.read_csv(DATASET_Y)
    print(y.head())
    print(y.shape)

    # pipeline design fix
    # X, y = X.iloc[:20000, :], y.iloc[:20000, :]

    # train XGBoost ML model for multiclass classification  
    model = XGBoostMulticlassClassifier(labels, '../models/model_xgboost_accuracy', use_instance_weights=False, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    model.train(X, y, scoring='accuracy')

    model = XGBoostMulticlassClassifier(labels, '../models/model_xgboost_accuracy_standardize', use_instance_weights=False, manual_weights=None, standardize=True, random_state=RANDOM_STATE)
    model.train(X, y, scoring='accuracy')
    
    model = XGBoostMulticlassClassifier(labels, '../models/model_xgboost_f1', use_instance_weights=False, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    model.train(X, y, scoring=make_scorer(f1_score, average='weighted'))

    model = XGBoostMulticlassClassifier(labels, '../models/model_xgboost_f1_standardize', use_instance_weights=False, manual_weights=None, standardize=True, random_state=RANDOM_STATE)
    model.train(X, y, scoring=make_scorer(f1_score, average='weighted'))

    weights = {
        0: 100,  # "BFS-EQ-16-0" 
        1: 100,  # "DFS-ES-16-0" 
        2: 5,    # "BFS-EQ-64-0"
        3: 100,  # "DFS-LL-0-0"
        4: 10,   # "DFS-ES-256-0"
        5: 10,   # "DFS-ES-64-0"
        6: 10,   # "BFS-EQ-256-0"
        7: 1,    # "BFS-AD-0-0"
        8: 20,   # "BFS-LL-0-0"
        9: 100   # "DFS-AD-0-0"
    }
    model = XGBoostMulticlassClassifier(labels, '../models/model_xgboost_manual_weights_accuracy', use_instance_weights=True, manual_weights=weights, standardize=False, random_state=RANDOM_STATE)
    model.train(X, y, scoring='accuracy')

    model = XGBoostMulticlassClassifier(labels, '../models/model_xgboost_manual_weights_f1', use_instance_weights=True, manual_weights=weights, standardize=False, random_state=RANDOM_STATE)
    model.train(X, y, scoring=make_scorer(f1_score, average='weighted'))

    model = XGBoostMulticlassClassifier(labels, '../models/model_xgboost_auto_weights_accuracy', use_instance_weights=True, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    model.train(X, y, scoring='accuracy')

    model = XGBoostMulticlassClassifier(labels, '../models/model_xgboost_auto_weights_accuracy_standardize', use_instance_weights=True, manual_weights=None, standardize=True, random_state=RANDOM_STATE)
    model.train(X, y, scoring='accuracy')

    model = XGBoostMulticlassClassifier(labels, '../models/model_xgboost_auto_weights_f1', use_instance_weights=True, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    model.train(X, y, scoring=make_scorer(f1_score, average='weighted'))

    model = XGBoostMulticlassClassifier(labels, '../models/model_xgboost_auto_weights_f1_standardize', use_instance_weights=True, manual_weights=None, standardize=True, random_state=RANDOM_STATE)
    model.train(X, y, scoring=make_scorer(f1_score, average='weighted'))


