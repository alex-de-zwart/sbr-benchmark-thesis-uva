FOLDERPATH = r'C:\Users\ZwartAlexde\Documents\UvA\thesis\thesis_sbr\data\prepared\vsknn'

import pandas as pd
from os import listdir
from os.path import isfile, join
from algorithms.vsknn.vsknn import VMContextKNN
from algorithms.vsknn.vsknn_parameters import VsknnParamSearch

from dataio.predictions import PredictionsWriter
from dataio.sessionloader.latency_writter import LatencyWriter
from dataio.sessionloader.loader import Loader
from algorithms.evaluation.replayer import Replayer
from algorithms.evaluation.utils import DataframeStatistics

from dataio.predictions import PredictionsReader
from dataio.sessionloader.loader import Loader
from algorithms.evaluation.metrics.coverage import Coverage
from algorithms.evaluation.metrics.popularity import Popularity
from algorithms.evaluation.metrics.accuracy import MRR, HitRate
from algorithms.evaluation.metrics.accuracy_multiple import Precision, Recall, MAP, NDCG


def run(training_csv_path, test_csv_path, params, outputfilename):
    print('reading training data')
    training_df = Loader.read_csv(training_csv_path)
    DataframeStatistics.print_df_statistics('training', training_df)

    print('reading test data')
    test_df = Loader.read_csv(test_csv_path)
    DataframeStatistics.print_df_statistics('test', test_df)

    print('model.fit(training_df) start')
    model = VMContextKNN(k=params["k"], sample_size=params["sample_size"], weighting=params["weighting"],
                         weighting_score=params["weighting_score"], idf_weighting=params["idf_weighting"])

    model.fit(training_df)
    print('model.fit(training_df) done')
    replayer = Replayer(test_df)

    predictions_writer = PredictionsWriter(outputfilename=outputfilename, evaluation_n=20)
    for (current_session_id, current_item_id, ts, rest) in replayer.next_sequence():

        recommendations = model.predict_next(session_id=current_session_id, input_item_id=current_item_id,
                                             timestamp=ts)

        predictions_writer.appendline(recommendations, rest)

    return training_df





def main(folder_path: str) -> None:
    """

    :param folder_path: the path where the training and test files are in
    :return: output from the run() function predictions
    """

    test = f'{folder_path}/item_views_train_valid.1.txt'
    train = f'{folder_path}/item_views_train_tr.1.txt'

    print(f'Loading train: {train}\n'
          f'Loading test: {test}')

    params = VsknnParamSearch.get_parameters()

    score_exp = []
    # score_exp.append("NDCG@20", "MAP@20", "Precision@20", "Recall@20", "HitRate@20", "MRR@20", "Coverage@20", "Popularity@20")
    for ele, param in enumerate(params):

        outputfilename = f'params/vsknn_py_predictions.{ele}.txt'
        training_df = \
            run(training_csv_path=train, test_csv_path=test, params=param, outputfilename=outputfilename)

        reader = PredictionsReader(inputfilename=outputfilename,
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

    cols = ['NDCG@20', 'MAP@20', 'Precision@20', 'Recall@20', 'HitRate@20', 'MRR@20', 'Coverage@20', 'Popularity@20']
    final = pd.DataFrame(score_exp, columns=cols)
    highest = final["MRR@20"].values.astype(float).argmax()

    print(f'Best hyper params settings are: {params[highest]} \n'
          f'With scores: {final.iloc[highest]}')


if __name__ == "__main__":
    main(FOLDERPATH)







