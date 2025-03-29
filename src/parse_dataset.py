import os 
import sys 
import json
import tqdm 
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

DATASET_CSV = '../data/dataset.csv'
DATASET_JSON = '../data/dataset.json'

"""
[
    {
        "name": "Harness.main(java.lang.String[])",
        "number of nodes": 20,
        "number of edges": 19,
        "depth": 9,
        "width": 3,
        "diameter": 10,
        "configuration": "BFS-EQ-16-0",
        "avg time": 1964.46875
    },
    {
        "name": "Harness.main(java.lang.String[])",
        "number of nodes": 20,
        "number of edges": 19,
        "depth": 9,
        "width": 3,
        "diameter": 10,
        "configuration": "DFS-ES-16-0",
        "avg time": 2967.59375
    },
    ...
]
"""

FEATURE_NAMES = ('number of nodes', 'number of edges', 'depth', 'width', 'diameter')
LABELS = ('BFS-EQ-16-0', 'DFS-ES-16-0', 'BFS-EQ-64-0', 'DFS-LL-0-0', 'DFS-ES-256-0', 'DFS-ES-64-0', 'BFS-EQ-256-0', 'BFS-AD-0-0', 'BFS-LL-0-0', 'DFS-AD-0-0')

DATASET_X = '../data/dataset_x.csv'
DATASET_Y = '../data/dataset_y.csv'
DATASET_CONFIG = '../data/dataset_config.json'

def parse_feature_vector(data: dict) -> list[int]:
    """
    Feature vector consisting of nodes, in that order: 
        "number of nodes": 20,
        "number of edges": 19,
        "depth": 9,
        "width": 3,
        "diameter": 10,
    """
    feature_value = []
    for feature_name in FEATURE_NAMES:
        assert feature_name in data, "Missing feature {} in data: {}".format(feature_name, data)
        feature_value.append(data[feature_name])

    return tuple(feature_value)

def plot_distribution(y, labels):
    """Plots the bin counts of numbers in the given array (0-9) with values on top."""
    counts = np.bincount(y, minlength=10)  # Ensure bins for 0-9
    
    plt.bar(range(len(labels)), counts, tick_label=labels)
    
    # Add value labels on top of each bar
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12)

    plt.xlabel("Number")
    plt.ylabel("Count")
    plt.title("Dataset Labels Distribution")
    plt.show()

if __name__ == "__main__":

    parsed_dataset = {}
    with open(DATASET_JSON, 'r') as f:
        dataset = json.load(f)
        
        for data in tqdm.tqdm(dataset):
            fname = data['name']
            fvector = parse_feature_vector(data)
            config = data['configuration']
            time = data['avg time']

            if (fname, fvector) not in parsed_dataset:
                parsed_dataset[(fname, fvector)] = []
            parsed_dataset[(fname, fvector)].append((config, time))


    ids, x, y = [], [], []
    for elem in tqdm.tqdm(parsed_dataset):
        sorted_traversals = sorted(parsed_dataset[elem], key=lambda runtime: runtime[1])
        
        ids.append(elem[0])
        x.append(elem[1])
        y.append(LABELS.index(sorted_traversals[0][0]))

        # print(elem, '---->', sorted_traversals)
        # print('Id:', ids[-1])
        # print('x:', x[-1])
        # print('y:', y[-1])
        # print('label:', sorted_traversals[0][0])
        # input('ENTER?')
    x = pd.DataFrame(x, columns=FEATURE_NAMES)
    y = pd.Series(y)

    # Plot stats
    print('Number of elements: ', len(x))
    x.describe().to_csv('../stats/features_stats.csv')
    plot_distribution(y, LABELS)

    # Save dataset
    print('Saving the dataset..')
    x.to_csv(DATASET_X, index=False)
    y.to_csv(DATASET_Y, index=False)
    with open(DATASET_CONFIG, 'w') as fp:
        json.dump({'labels': LABELS, 'ids': ids}, fp)
    
