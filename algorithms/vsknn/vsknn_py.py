FOLDERPATH = '/home/ec2-user/SageMaker/sb-rec-system/data/prepared/vsknn'

import sys
module_path = "/home/ec2-user/SageMaker/sb-rec-system"
if module_path not in sys.path:
    sys.path.append(module_path)

from os import listdir
from os.path import isfile, join
from algorithms.vsknn.vsknn import VMContextKNN

from dataio.predictions import PredictionsWriter
from dataio.sessionloader.latency_writter import LatencyWriter
from dataio.sessionloader.loader import Loader
from algorithms.evaluation.replayer import Replayer
from algorithms.evaluation.utils import DataframeStatistics


def run(training_csv_path, test_csv_path, outputfilename, latency_outputfilename):
    print('reading training data')
    training_df = Loader.read_csv(training_csv_path)
    DataframeStatistics.print_df_statistics('training', training_df)

    print('reading test data')
    test_df = Loader.read_csv(test_csv_path)
    DataframeStatistics.print_df_statistics('test', test_df)

    test_limit = 1000000
    if len(test_df) > test_limit:
        # python is slow at scale thus we test on a small subset of testdata
        print('reducing testdata to ' + str(test_limit))
        test_df = test_df.head(test_limit)
        DataframeStatistics.print_df_statistics('reduced test data', test_df)

    print('model.fit(training_df) start')
    model = VMContextKNN(k=1500, sample_size=500, weighting='quadratic', weighting_score='div', idf_weighting=1)
    model.fit(training_df)
    print('model.fit(training_df) done')
    replayer = Replayer(test_df)

    predictions_writer = PredictionsWriter(outputfilename=outputfilename, evaluation_n=20)
    latency_writer = LatencyWriter(latency_outputfilename)
    for (current_session_id, current_item_id, ts, rest) in replayer.next_sequence():

        recommendations = model.predict_next(session_id=current_session_id, input_item_id=current_item_id,
                                             timestamp=ts)

        predictions_writer.appendline(recommendations, rest)

        if len(model.get_latencies()) < 10000:
            if len(model.get_latencies()) % 1000 == 0:
                print('qty predictions:' + str(len(model.get_latencies())))
        else:
            if len(model.get_latencies()) % 10000 == 0:
                print('qty predictions:' + str(len(model.get_latencies())))

    predictions_writer.close()
    for (position, latency) in model.get_latencies():
        latency_writer.append_line(position, latency)
    latency_writer.close()


def main(folder_path: str) -> None:
    """

    :param folder_path: the path where the training and test files are in
    :return: output from the run() function latencys and predictions
    """

    files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    n_slices = sum('test' in s for s in files)

    for ele in range(n_slices):
        test = f'{folder_path}/item_views_test.{ele}.txt'
        train = f'{folder_path}/item_views_train.{ele}.txt'

        print(f'Loading train: {train}\n'
              f'Loading test: {test}')

        outputfilename = f'predictions/vsknn_py_predictions.{ele}.txt'
        latency_outputfilename = f'latencies/latency.{ele}.txt'

        run(training_csv_path=train, test_csv_path=test, outputfilename=outputfilename,
            latency_outputfilename=latency_outputfilename)



if __name__ == '__main__':
    main(folder_path=FOLDERPATH)
