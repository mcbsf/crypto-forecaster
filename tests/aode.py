import codecs

import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold

from .util import context
from .util.data_collection import DataCollection, EDataType

from forecaster.classifiers import AODE


print('Collecting data...')
data_collection = DataCollection('res/promise/')

aode = AODE()
kf = KFold(n_splits=5, shuffle=True) # 80% for training, 20% for testing

results = {}
estimation = 'count'
for relation in data_collection.documents:
    file_ = codecs.open('res/promise/results/' + relation + '-' + estimation + '.txt', 'w+', 'utf-8')

    results[relation] = {}

    print('\nTesting for ' + relation + ' data set')
    relation_data, relation_labels = data_collection.get_data_label(relation)
    data_len = len(relation_data)
    features_amount = relation_data[0].size
    print('\tTotal data collected: ' + str(data_len))
    print('\tTotal of features per data: ' + str(features_amount))

    # Only numerical features
    features_types = data_collection.get_features_types(relation)

    metrics = []
    all_precisions = []
    all_recall = []
    all_f1 = []

    for train_indices, test_indices in kf.split(relation_data):
        train_data, test_data = relation_data[train_indices], relation_data[test_indices]
        train_labels, test_labels = relation_labels[train_indices], relation_labels[test_indices]

        # Training
        aode.fit(train_data, train_labels, online=False)

        # Test
        pred_labels = []
        for data in test_data:
            pred_labels.append(aode.predict(data, estimation=estimation))

        metrics = precision_recall_fscore_support(test_labels, pred_labels, average='weighted')
        precision, recall, f1, _ = metrics

        all_precisions.append(np.mean(precision))
        all_recall.append(np.mean(recall))
        all_f1.append(np.mean(f1))

        results[relation]['precision'] = all_precisions
        results[relation]['recall'] = all_recall
        results[relation]['f1'] = all_f1

    file_.write('Precision:\n' + str(all_precisions) + '\nMean: ' + str(np.mean(all_precisions)))
    file_.write('\n\nRecall:\n' + str(all_recall) + '\nMean: ' + str(np.mean(all_recall)))
    file_.write('\n\nF1:\n' + str(all_f1) + '\nMean: ' + str(np.mean(all_f1)))
