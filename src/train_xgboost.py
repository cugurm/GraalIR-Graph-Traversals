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

DATASET_X = '../data_v4/dataset_x.csv'
DATASET_Y = '../data_v4/dataset_y.csv'
DATASET_Y_BINARY = '../data_v4/dataset_y_binary.csv'
DATASET_CONFIG = '../data_v4/dataset_config.json'
RANDOM_STATE = 1234
MODELS_DIR = '../models_v4/'


class XGBoostMulticlassClassifier:
    def __init__(self, labels, feature_names, training_dir, use_instance_weights=False, manual_weights=None, standardize=False, random_state=RANDOM_STATE):
        self.labels = labels
        self.feature_names = feature_names

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
        sample_weights_train, sample_weights_test = None, None
        if self.use_instance_weights:
            if self.manual_weights:
                sample_weights_train = np.array([self.manual_weights[label] for label in y_train.iloc[:, 0]])
                sample_weights_test = np.array([self.manual_weights[label] for label in y_test.iloc[:, 0]])
            else:
                class_weights = self._compute_class_weights(y_train)
                sample_weights_train = np.array([class_weights[int(label)] for label in y_train.iloc[:, 0]])
                sample_weights_test = np.array([class_weights[int(label)] for label in y_test.iloc[:, 0]])

        # Perform hyperparameter tuning if enabled
        best_params = self.tune_hyperparameters(X_train, y_train, sample_weights_train, scoring) if use_grid_search else {}

        # Train final model
        self.model = self._define_model(best_params)
        if self.standardize:
            self.model.fit(X_train, y_train, model__sample_weight=sample_weights_train)
        else:
            self.model.fit(X_train, y_train, sample_weight=sample_weights_train)

        # Evaluate model
        self.evaluate(X_train, y_train, 'training', sample_weights_train)
        self.evaluate(X_test, y_test, 'test', sample_weights_test)

        # Plot feature importance
        XGBoostMulticlassClassifier.analyse_model(self.model, self.feature_names, self.training_dir, plot_n=25)

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
                'model__max_depth': [7, 10, 15],
                'model__n_estimators': [200, 250, 300],
            }
        else:
            param_grid = {
                'max_depth': [20],
                'n_estimators': [2000],
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
        
        # Save best configuration
        config_path = os.path.join(self.training_dir, 'best_params') 
        with open(config_path, 'w') as fp:
            json.dump(grid_search.best_params_, fp)
        print('Saved best parameters to: {}'.format(config_path))
        
        return grid_search.best_params_

    def evaluate(self, X, y, dataset_name, instance_weights=None):
        if self.model is None:
            raise ValueError("Model is not trained. Call model.train() first.")

        y_pred = self.model.predict(X)

        # === Unweighted metrics ===
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        report = classification_report(
            y, y_pred,
            target_names=[str(label) for label in self.labels],
            zero_division=0
        )
        conf_matrix = confusion_matrix(y, y_pred)

        # === Weighted metrics (if instance_weights is provided) ===
        if instance_weights is not None:
            weighted_accuracy = accuracy_score(y, y_pred, sample_weight=instance_weights)
            weighted_precision = precision_score(y, y_pred, average='weighted', zero_division=0, sample_weight=instance_weights)
            weighted_recall = recall_score(y, y_pred, average='weighted', zero_division=0, sample_weight=instance_weights)
            weighted_f1 = f1_score(y, y_pred, average='weighted', zero_division=0, sample_weight=instance_weights)
            weighted_report = classification_report(
                y, y_pred,
                target_names=[str(label) for label in self.labels],
                zero_division=0,
                sample_weight=instance_weights
            )
        else:
            weighted_accuracy = weighted_precision = weighted_recall = weighted_f1 = None
            weighted_report = None

        # === Save logs ===
        log_data = {
            "dataset": dataset_name,
            "unweighted": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": conf_matrix.tolist()
            },
        }

        if instance_weights is not None:
            log_data["weighted"] = {
                "accuracy": weighted_accuracy,
                "precision": weighted_precision,
                "recall": weighted_recall,
                "f1_score": weighted_f1,
            }

        log_path = os.path.join(self.training_dir, f"{dataset_name}_training_log.json")
        report_path = os.path.join(self.training_dir, f"{dataset_name}_classification_report.txt")
        weighted_report_path = os.path.join(self.training_dir, f"{dataset_name}_weighted_classification_report.txt")

        with open(log_path, "w") as log_file:
            json.dump(log_data, log_file, indent=4)

        with open(report_path, "w") as report_file:
            report_file.write(report)

        if instance_weights is not None:
            with open(weighted_report_path, "w") as report_file:
                report_file.write(weighted_report)

        # === Print logs ===
        print(f'[{dataset_name.upper()}] Unweighted Accuracy: {accuracy:.4f}')
        print(f'[{dataset_name.upper()}] Unweighted Precision: {precision:.4f}')
        print(f'[{dataset_name.upper()}] Unweighted Recall: {recall:.4f}')
        print(f'[{dataset_name.upper()}] Unweighted F1 Score: {f1:.4f}')
        print(f'[{dataset_name.upper()}] Unweighted Classification Report:')
        print(report)

        if instance_weights is not None:
            print(f'[{dataset_name.upper()}] Weighted Accuracy: {weighted_accuracy:.4f}')
            print(f'[{dataset_name.upper()}] Weighted Precision: {weighted_precision:.4f}')
            print(f'[{dataset_name.upper()}] Weighted Recall: {weighted_recall:.4f}')
            print(f'[{dataset_name.upper()}] Weighted F1 Score: {weighted_f1:.4f}')
            print(f'[{dataset_name.upper()}] Weighted Classification Report:')
            print(weighted_report)
   
        # Plot Confusion Matrix
        matrix_path = os.path.join(self.training_dir, f"{dataset_name}_confusion_matrix.png")
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix ({dataset_name})')
        plt.tight_layout()
        plt.savefig(matrix_path)

    @staticmethod
    def analyse_model(model, feature_names, output_dir, plot_n=25):
        """
        Extracts feature importance from a trained XGBoost model and saves it to a CSV file.

        The extracted feature importance metrics:
            - gain: Average gain of splits where the feature is used (higher is better).
            - cover: Average coverage (number of samples affected) of splits where the feature is used.
            - total_gain: Total gain across all splits involving the feature.
            - total_cover: Total coverage across all splits involving the feature.
            - weight: Number of times a feature is used in splits.

        Parameters:
            - model: Trained XGBoost model (XGBClassifier or XGBRegressor)
        """
        # dump feature importance
        booster = model.get_booster()
        booster.feature_names = feature_names
        importance_types = ['gain', 'cover', 'total_gain', 'total_cover', 'weight']
        importance_data = {feature: {} for feature in booster.feature_names}

        for importance_type in importance_types:
            importance_dict = booster.get_score(importance_type=importance_type)
            for feature, value in importance_dict.items():
                importance_data.setdefault(feature, {})[importance_type] = value

        output_path = os.path.join(output_dir, 'feature_importance.csv')
        importance_df = pd.DataFrame.from_dict(importance_data, orient='index').fillna(0)
        importance_df.index.name = 'Feature'
        importance_df.reset_index(inplace=True)
        importance_df['Feature'] = importance_df['Feature'].apply(lambda x: booster.feature_names[int(x[1:])] if x[1:].isdigit() else x)
        importance_df = importance_df.sort_values(by='gain', ascending=False)
        importance_df.to_csv(output_path, index=False)
        print('Feature importance saved to {}'.format(output_path))

        # plot gini index
        plot_path = os.path.join(output_dir, 'feature_importance.png')
        assert plot_n < len(booster.feature_names), 'Can not plot gini index for more than {} features.'.format(len(booster.feature_names))
        top_features = importance_df.head(plot_n)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(top_features['Feature'], top_features['gain'], color='skyblue')
        plt.xlabel('Gain')
        plt.ylabel('Feature')
        plt.title('Top {} Most Important Features (Gain)'.format(plot_n))
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization
        for bar, value in zip(bars, top_features['gain']):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, '{:.2f}'.format(value), va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        print("Feature importance plot saved to {}".format(plot_path))


if __name__ == "__main__":

    # load data
    print('Loading the dataset configuration..')
    with open(DATASET_CONFIG, 'r') as fp:
        config = json.load(fp)
        labels = config['labels']
        labels_binary = config['labels binary']
        feature_names = config['feature names']
        ids = config['ids']

        print("Labels: {}".format(labels))
        print("Labels binary: {}".format(labels_binary))
        print("Number of methods (ids): {}".format(len(ids)))

    print('Loading feature set..')
    X = pd.read_csv(DATASET_X)
    print(X.head())
    print(X.shape)

    print('Loading labels..')
    y = pd.read_csv(DATASET_Y)
    print(y.head())
    print(y.shape)

    y_binary = pd.read_csv(DATASET_Y_BINARY)
    print(y_binary.head())
    print(y_binary.shape)

    # pipeline design fix
    # param_grid = {
    #     'max_depth': [10, 20, 30],
    #     'n_estimators': [500, 1000, 2000, 3000, 5000],
    # }
    # X, y = X.iloc[:20000, :], y.iloc[:20000, :]

    # train XGBoost ML model for multiclass classification  
    # model = XGBoostMulticlassClassifier(labels, os.path.join(MODELS_DIR, 'model_xgboost_accuracy'), use_instance_weights=False, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y, scoring='accuracy')

    # model = XGBoostMulticlassClassifier(labels, os.path.join(MODELS_DIR, 'model_xgboost_accuracy_standardize'), use_instance_weights=False, manual_weights=None, standardize=True, random_state=RANDOM_STATE)
    # model.train(X, y, scoring='accuracy')
    
    # model = XGBoostMulticlassClassifier(labels, os.path.join(MODELS_DIR, 'model_xgboost_f1'), use_instance_weights=False, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y, scoring=make_scorer(f1_score, average='weighted'))

    # model = XGBoostMulticlassClassifier(labels, os.path.join(MODELS_DIR, 'model_xgboost_f1_standardize'), use_instance_weights=False, manual_weights=None, standardize=True, random_state=RANDOM_STATE)
    # model.train(X, y, scoring=make_scorer(f1_score, average='weighted'))

    # weights = {
    #     0: 100,  # "BFS-EQ-16-0" 
    #     1: 100,  # "DFS-ES-16-0" 
    #     2: 5,    # "BFS-EQ-64-0"
    #     3: 100,  # "DFS-LL-0-0"
    #     4: 10,   # "DFS-ES-256-0"
    #     5: 10,   # "DFS-ES-64-0"
    #     6: 10,   # "BFS-EQ-256-0"
    #     7: 1,    # "BFS-AD-0-0"
    #     8: 20,   # "BFS-LL-0-0"
    #     9: 100   # "DFS-AD-0-0"
    # }
    # model = XGBoostMulticlassClassifier(labels, os.path.join(MODELS_DIR, 'model_xgboost_manual_weights_accuracy'), use_instance_weights=True, manual_weights=weights, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y, scoring='accuracy')

    # model = XGBoostMulticlassClassifier(labels, os.path.join(MODELS_DIR, 'model_xgboost_manual_weights_f1'), use_instance_weights=True, manual_weights=weights, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y, scoring=make_scorer(f1_score, average='weighted'))

    # model = XGBoostMulticlassClassifier(labels, os.path.join(MODELS_DIR, 'model_xgboost_auto_weights_accuracy'), use_instance_weights=True, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y, scoring='accuracy')

    # model = XGBoostMulticlassClassifier(labels, os.path.join(MODELS_DIR, 'model_xgboost_auto_weights_accuracy_standardize_extended_grid'), use_instance_weights=True, manual_weights=None, standardize=True, random_state=RANDOM_STATE)
    # model.train(X, y, scoring='accuracy')

    # model = XGBoostMulticlassClassifier(labels, os.path.join(MODELS_DIR, 'model_xgboost_auto_weights_f1'), use_instance_weights=True, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y, scoring=make_scorer(f1_score, average='weighted'))

    # model = XGBoostMulticlassClassifier(labels, os.path.join(MODELS_DIR, 'model_xgboost_auto_weights_f1_standardize_extended_grid'), use_instance_weights=True, manual_weights=None, standardize=True, random_state=RANDOM_STATE)
    # model.train(X, y, scoring=make_scorer(f1_score, average='weighted'))

    # model = XGBoostMulticlassClassifier(labels, os.path.join(MODELS_DIR, 'model_xgboost_2000x20'), use_instance_weights=True, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y, scoring=make_scorer(f1_score, average='weighted'))

    ##################################################################################################################################################
    # Train XGBoost model for binary classification

    # param_grid = {
    #     'max_depth': [20],
    #     'n_estimators': [2000],
    # }
    # Computed class weights: {0: 0.5541467528816418, 1: 5.117082035306335}
    # model = XGBoostMulticlassClassifier(labels_binary, feature_names, os.path.join(MODELS_DIR, 'model_xgboost_2000x20_binary'), use_instance_weights=True, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y_binary, scoring=make_scorer(f1_score, average='weighted'))

    # param_grid = {
    #     'max_depth': [20],
    #     'n_estimators': [2000],
    # }
    # model = XGBoostMulticlassClassifier(labels_binary, feature_names, os.path.join(MODELS_DIR, 'model_xgboost_2000x20_binary_manual_weights'), use_instance_weights=True, manual_weights={0: 0.5, 1: 10}, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y_binary, scoring=make_scorer(f1_score, average='weighted'))

    # param_grid = {
    #     'max_depth': [20],
    #     'n_estimators': [2000],
    # }
    # model = XGBoostMulticlassClassifier(labels_binary, feature_names, os.path.join(MODELS_DIR, 'model_xgboost_2000x20_binary_manual_weights_25'), use_instance_weights=True, manual_weights={0: 0.5, 1: 25}, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y_binary, scoring=make_scorer(f1_score, average='weighted'))

    # param_grid = {
    #     'max_depth': [20],
    #     'n_estimators': [2000],
    # }
    # model = XGBoostMulticlassClassifier(labels_binary, feature_names, os.path.join(MODELS_DIR, 'model_xgboost_2000x20_binary_manual_weights_15'), use_instance_weights=True, manual_weights={0: 0.5, 1: 15}, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y_binary, scoring=make_scorer(f1_score, average='weighted'))

    # param_grid = {
    #     'max_depth': [20],
    #     'n_estimators': [2000],
    # }
    model = XGBoostMulticlassClassifier(labels_binary, feature_names, os.path.join(MODELS_DIR, 'model_xgboost_2000x20_binary_manual_weights_13'), use_instance_weights=True, manual_weights={0: 0.5, 1: 13}, standardize=False, random_state=RANDOM_STATE)
    model.train(X, y_binary, scoring=make_scorer(f1_score, average='weighted'))

    # TODO: accuracy
    # param_grid = {
    #     'max_depth': [20],
    #     'n_estimators': [2000],
    # }
    # model = XGBoostMulticlassClassifier(labels_binary, feature_names, os.path.join(MODELS_DIR, 'model_xgboost_2000x20_binary_accuracy'), use_instance_weights=True, manual_weights=False, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y_binary, scoring='accuracy')

    # param_grid = {
    #     'max_depth': [10, 20, 30],
    #     'n_estimators': [500, 1000, 2000, 3000, 5000],
    # }
    # model = XGBoostMulticlassClassifier(labels_binary, feature_names, os.path.join(MODELS_DIR, 'model_xgboost_binary_autoweights_large'), use_instance_weights=True, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y_binary, scoring=make_scorer(f1_score, average='weighted'))

    # param_grid = {
    #     'max_depth': [3, 5, 7, 10, 12],
    #     'n_estimators': [100, 200, 300, 500, 700],
    # }
    # model = XGBoostMulticlassClassifier(labels_binary, feature_names, os.path.join(MODELS_DIR, 'model_xgboost_binary_autoweights_small'), use_instance_weights=True, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y_binary, scoring=make_scorer(f1_score, average='weighted'))

    # param_grid = {
    #     'max_depth': [2, 3, 5, 7],
    #     'n_estimators': [10, 20, 50, 100, 200, 300, 500, 700],
    # }
    # model = XGBoostMulticlassClassifier(labels_binary, feature_names, os.path.join(MODELS_DIR, 'model_xgboost_binary_autoweights_mini'), use_instance_weights=True, manual_weights=None, standardize=False, random_state=RANDOM_STATE)
    # model.train(X, y_binary, scoring=make_scorer(f1_score, average='weighted'))




