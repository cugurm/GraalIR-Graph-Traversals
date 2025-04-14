import os 
import sys 
import json
import tqdm 
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

DATASET_CSV = '../data_v3/dataset.csv'
DATASET_JSON = '../data_v3/dataset.json'
STATS_DIR = '../stats_v3/'

"""
[
 {
        "name": "Harness.main(java.lang.String[])",
        "number of nodes": 20,
        "number of edges": 19,
        "depth": 9,
        "width": 3,
        "diameter": 10,
        "number of binary splits": 9,
        "number of non binary splits": 0,
        "total number of splits": 9,
        "max cardinality": 2,
        "avg cardinality including binary splits": 2.0,
        "avg cardinality excluding binary splits": 0.0,
        "ratio nodes edges": 1.0526315789473684,
        "ratio edges nodes": 0.95,
        "min degree": 2,
        "max degree": 3,
        "avg degree": 2.888888888888889,
        "coefficient variation degrees": 0.10878565864408425,
        "entropy of distributions degrees": 0.5032583347756457,
        "min in degree": 1,
        "max in degree": 1,
        "avg in degree": 1.0,
        "coefficient variation in degrees": 0.0,
        "entropy of distributions in degrees": 0.0,
        "min out degree": 1,
        "max out degree": 2,
        "avg out degree": 1.9,
        "coefficient variation out degrees": 0.15789473684210525,
        "entropy of distributions out degrees": 0.46899559358928117,
        "configuration": "BFS-EQ-16-0",
        "avg time": 1964.46875
    },
    ...
]
"""

FEATURE_NAMES = ('number of nodes', 'number of edges', 'depth', 'width', 'diameter', 'number of binary splits', 'number of non binary splits', 'total number of splits', 
                 'max cardinality', 'avg cardinality including binary splits', 'avg cardinality excluding binary splits', 
                 'ratio nodes edges', 'ratio edges nodes', 'min degree', 'max degree', 'avg degree', 'coefficient variation degrees', 'entropy of distributions degrees', 
                 'min in degree', 'max in degree', 'avg in degree', 'coefficient variation in degrees', 'entropy of distributions in degrees', 
                 'min out degree', 'max out degree', 'avg out degree', 'coefficient variation out degrees', 'entropy of distributions out degrees')
LABELS = ('BFS-EQ-16-0', 'DFS-ES-16-0', 'BFS-EQ-64-0', 'DFS-LL-0-0', 'DFS-ES-256-0', 'DFS-ES-64-0', 'BFS-EQ-256-0', 'BFS-AD-0-0', 'BFS-LL-0-0', 'DFS-AD-0-0')

DATASET_X = '../data_v3/dataset_x.csv'
DATASET_Y = '../data_v3/dataset_y.csv'
DATASET_CONFIG = '../data_v3/dataset_config.json'

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
    configs = set()
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
            configs.add(config)

        assert configs == set(LABELS), 'Invalid labels specified: {}. Collected labels: {}'.format(LABELS, configs)


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
    x.describe().to_csv(os.path.join(STATS_DIR, 'features_stats.csv'))
    plot_distribution(y, LABELS)

    # Save dataset
    print('Saving the dataset..')
    x.to_csv(DATASET_X, index=False)
    y.to_csv(DATASET_Y, index=False)
    with open(DATASET_CONFIG, 'w') as fp:
        json.dump({'labels': LABELS, 'ids': ids}, fp)
    
