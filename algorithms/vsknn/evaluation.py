import sys
module_path = "/home/ec2-user/SageMaker/sb-rec-system"
if module_path not in sys.path:
    sys.path.append(module_path)

from dataio.predictions import PredictionsReader
from dataio.sessionloader.loader import Loader
from algorithms.evaluation.metrics.coverage import Coverage
from algorithms.evaluation.metrics.popularity import Popularity
from algorithms.evaluation.metrics.accuracy import MRR, HitRate
from algorithms.evaluation.metrics.accuracy_multiple import Precision, Recall, MAP, NDCG

from os import listdir
from os.path import isfile, join
import pandas as pd

TRAIN_PATH = '/home/ec2-user/SageMaker/sb-rec-system/data/prepared/vsknn'
PRED_PATH = '/home/ec2-user/SageMaker/sb-rec-system/algorithms/vsknn/predictions'


if __name__ == '__main__':

    train_files = [f for f in listdir(TRAIN_PATH) if isfile(join(TRAIN_PATH, f))]
    n_slices = sum('item_views_train.' in s for s in train_files)

    score_exp = []
    for ele in range(n_slices):
        train = f'{TRAIN_PATH}/item_views_train.{ele}.txt'
        pred = f'predictions/vsknn_py_predictions.{ele}.txt'

        training_df = Loader.read_csv(train)

        reader = PredictionsReader(inputfilename=pred,
                                   training_df=training_df)

        metrics = [NDCG(), MAP(), Precision(), Recall(), HitRate(), MRR(), Coverage(training_df=training_df),
                   Popularity(training_df=training_df)]

        for (recommendations, next_items) in reader.get_next_line():
            for metric in metrics:
                metric.add(recommendations, next_items)
        scores = []
        for metric in metrics:
            metric_name, score = metric.result()
            scores.append("%.4f" % score)
            print(metric_name, "%.4f" % score)

        score_exp.append(scores)

    score_exp = pd.DataFrame(score_exp)
    cols = ['NDCG@20', 'MAP@20', 'Precision@20', 'Recall@20', 'HitRate@20', 'MRR@20', 'Coverage@20', 'Popularity@20']
    score_exp.to_csv('model_performance_over_all_slices.csv', sep=';', decimal=",", header=cols, index=0)
