import os 
import sys 
import json
import tqdm 
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

DATASET_CSV = '../data_v4/dataset.csv'
DATASET_JSON = '../data_v4/dataset.json'
STATS_DIR = '../stats_v4/'

"""
[
    {
        "name": "Harness.main(java.lang.String[])",
        "number of nodes": 20,
        "number of edges": 27,
        "depth": 9,
        "width": 4,
        "number of binary splits": 9,
        "number of non binary splits": 0,
        "total number of splits": 9,
        "max cardinality": 2,
        "avg cardinality including binary splits": 2.0,
        "avg cardinality excluding binary splits": 0.0,
        "ratio nodes edges": 0.7407407407407407,
        "ratio edges nodes": 1.35,
        "min degree": 1,
        "max degree": 9,
        "avg degree": 2.736842105263158,
        "coefficient variation degrees": 0.5788429593574004,
        "entropy of distributions degrees": 1.4832261969709113,
        "min in degree": 1,
        "max in degree": 9,
        "avg in degree": 1.4210526315789473,
        "coefficient variation in degrees": 1.257078722109418,
        "entropy of distributions in degrees": 0.2974722489192897,
        "min out degree": 0,
        "max out degree": 2,
        "avg out degree": 1.35,
        "coefficient variation out degrees": 0.48432210483785254,
        "entropy of distributions out degrees": 1.3689955935892812,
        "configuration": "BFS-EQ-16-0",
        "avg time": 1964.46875
    },
    ...
]
"""

FEATURE_NAMES = ('number of nodes', 'number of edges', 'depth', 'width', 'number of binary splits', 'number of non binary splits', 'total number of splits', 
                 'max cardinality', 'avg cardinality including binary splits', 'avg cardinality excluding binary splits', 
                 'ratio nodes edges', 'ratio edges nodes', 'min degree', 'max degree', 'avg degree', 'coefficient variation degrees', 'entropy of distributions degrees', 
                 'min in degree', 'max in degree', 'avg in degree', 'coefficient variation in degrees', 'entropy of distributions in degrees', 
                 'min out degree', 'max out degree', 'avg out degree', 'coefficient variation out degrees', 'entropy of distributions out degrees')
LABELS = ('BFS-EQ-16-0', 'DFS-ES-16-0', 'BFS-EQ-64-0', 'DFS-LL-0-0', 'DFS-ES-256-0', 'DFS-ES-64-0', 'BFS-EQ-256-0', 'BFS-AD-0-0', 'BFS-LL-0-0', 'DFS-AD-0-0')
LABELS_BINARY = ('BFS', 'DFS')

DATASET_X = '../data_v4/dataset_x.csv'
DATASET_Y = '../data_v4/dataset_y.csv'
DATASET_Y_BINARY = '../data_v4/dataset_y_binary.csv'
DATASET_CONFIG = '../data_v4/dataset_config.json'

def parse_feature_vector(data: dict) -> list[int]:
    """
    Feature vector consisting of nodes, in that order: 
        "number of nodes": 20,
        "number of edges": 19,
        "depth": 9,
        "width": 3,
        "diameter": 10, ...
    """
    feature_value = []
    for feature_name in FEATURE_NAMES:
        assert feature_name in data, "Missing feature {} in data: {}".format(feature_name, data)
        feature_value.append(data[feature_name])

    return tuple(feature_value)

def plot_distribution(y, labels, out_path, minlength):
    plt.figure(figsize=(14, 8)) 
    
    counts = np.bincount(y, minlength=minlength) 
    
    plt.bar(range(len(labels)), counts, tick_label=labels)
    
    # Add value labels on top of each bar
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12)

    plt.xlabel("Number")
    plt.ylabel("Count")
    plt.title("Dataset Labels Distribution")
    plt.savefig(out_path)
    plt.clf()

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


    ids, x, y, y_binary = [], [], [], []
    for elem in tqdm.tqdm(parsed_dataset):
        sorted_traversals = sorted(parsed_dataset[elem], key=lambda runtime: runtime[1])
        
        ids.append(elem[0])
        x.append(elem[1])
        label = sorted_traversals[0][0]
        y.append(LABELS.index(label))
        y_binary.append(LABELS_BINARY.index(label[0:3]))

        # print(elem, '---->', sorted_traversals)
        # print('Id:', ids[-1])
        # print('x:', x[-1])
        # print('y:', y[-1])
        # print('y_binary:', y_binary[-1])
        # print('label:', label)
        # input('ENTER?')
    
    # Create data
    x = pd.DataFrame(x, columns=FEATURE_NAMES)
    y = pd.Series(y)
    y_binary = pd.Series(y_binary)

    # Print data
    print('Collected data:')
    print(x.head())
    print(y.head())
    print(y_binary.head())

    # Plot stats
    print('Number of elements: ', len(x))
    x.describe().T.to_csv(os.path.join(STATS_DIR, 'features_stats.csv'))
    plot_distribution(y, LABELS, os.path.join(STATS_DIR, 'y_distribution.png'), minlength=10)
    plot_distribution(y_binary, LABELS_BINARY, os.path.join(STATS_DIR, 'y_binary_distribution.png'), minlength=2)

    # Save dataset
    print('Saving the dataset..')
    x.to_csv(DATASET_X, index=False)
    y.to_csv(DATASET_Y, index=False)
    y_binary.to_csv(DATASET_Y_BINARY, index=False)
    with open(DATASET_CONFIG, 'w') as fp:
        json.dump({'labels': LABELS, 'labels binary': LABELS_BINARY, 'ids': ids}, fp)
    
