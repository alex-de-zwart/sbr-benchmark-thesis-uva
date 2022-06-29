import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# DATA_PATH = '/home/ec2-user/SageMaker/sb-rec-system/data/raw/'
DATA_PATH = 's3://alex-private-aws-s3/sbr-system/'
DATA_PATH_PROCESSED = '/home/ec2-user/SageMaker/sb-rec-system//data/prepared/vsknn/'
DATA_FILE = 'item_views'

#max session length is 14 according to the 99th percentile
# filtering config (all methods)
MIN_SESSION_LENGTH = 2
MAX_SESSION_LENGTH = 14
MIN_ITEM_SUPPORT = 5

# min date config
MIN_DATE = '2021-06-01'

# days test default config
DAYS_TEST = 7

# slicing default config
NUM_SLICES = 3
DAYS_OFFSET = 0
DAYS_SHIFT = 71
DAYS_TRAIN = 64

# preprocessing to create data slices with a window
def preprocess_slices(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                      min_session_length=MIN_SESSION_LENGTH,
                      num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT, days_train=DAYS_TRAIN,
                      days_test=DAYS_TEST):
    print('start preprocess_slices..')
    data = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length)
    slice_data(data, path_proc + file, num_slices, days_offset, days_shift, days_train, days_test)


def load_data(file):
    print('start load_data..')
    # load csv
    data = pd.read_csv(file + '.csv', sep=',', header=None, skiprows=1, usecols=[0, 1, 3])
    # specify header names
    data.columns = ['TimeStr', 'SessionId', 'ItemId']

    # convert time string to timestamp and remove the original column
    data['Time'] = data.TimeStr.apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f').timestamp())  # This is not UTC. It does not really matter.
    del (data['TimeStr'])
    data = data[['SessionId','ItemId','Time']]

#     #test to convert itemids to numerical values
#     items = pd.Series(data.ItemId, dtype='category')
#     item_to_number = dict(zip(items, items.cat.codes))
#     data = data.replace({"ItemId": item_to_number})

    sesid = pd.Series(data.SessionId, dtype='category')
    ses_to_number = dict(zip(sesid, sesid.cat.codes))
    new = pd.DataFrame()
    new['SessionId'] = ses_to_number.keys()
    new['Session_renumbered'] = ses_to_number.values()

    data = data.merge(new, on=['SessionId'])
    data = data[['Session_renumbered', 'ItemId', 'Time']]
    data.columns = ['SessionId','ItemId','Time']
    
    #remove rows where the item is the same as the previous item
    data = data.groupby((data["ItemId"] != data["ItemId"].shift()).cumsum().values).first()

    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print(f"Loaded data set\n\tEvents: {len(data)}\n\tSessions: {data.SessionId.nunique()}\n\tItems: {data.ItemId.nunique()}"
          f"\n\tSpan: {data_start.date().isoformat()} / {data_end.date().isoformat()}\n\n")
    
#     with open('reversed_item_dict_vsknn.txt', 'w') as f:
#         print(item_to_number, file=f)

    return data


def filter_data(data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                max_session_length=MAX_SESSION_LENGTH):
    print('start filter_data..')

    # filter session length
    agg = data.groupby('SessionId').size().reset_index(name='count_view')
    session_lengths = agg[(agg['count_view'] >= min_session_length) & (agg['count_view'] <= max_session_length)]['SessionId'].tolist()
    data = data[data.SessionId.isin(session_lengths)]

    # filter item support
    item_supports = data.groupby('ItemId').size().reset_index(name='count')
    items = item_supports[item_supports['count'] >= min_item_support]['ItemId'].tolist()
    data = data[data['ItemId'].isin(items)]

    # filter session length
    agg = data.groupby('SessionId').size().reset_index(name='count_view')
    session_lengths = agg[(agg['count_view'] >= min_session_length) & (agg['count_view'] <= max_session_length)]['SessionId'].tolist()
    data = data[data.SessionId.isin(session_lengths)]

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    data.to_csv(DATA_PATH + 'cleaned_complete_dataset_check' + '.txt', sep='\t', index=False)

    print(f"Filtered data set\n\tEvents: {len(data)}\n\tSessions: {data.SessionId.nunique()}\n\tItems: "
          f"{data.ItemId.nunique()}\n\tSpan: {data_start.date().isoformat()} / {data_end.date().isoformat()}\n\n")

    return data


def slice_data(data, output_file, num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT,
               days_train=DAYS_TRAIN, days_test=DAYS_TEST):
    print('start slice_data..')
    for slice_id in range(0, num_slices):
        split_data_slice(data, output_file, slice_id, days_offset + (slice_id * days_shift), days_train, days_test)


def split_data_slice(data, output_file, slice_id, days_offset, days_train, days_test):
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print(f"Full data set {slice_id}\n\tEvents: {len(data)}\n\tSessions: {data.SessionId.nunique()}\n\tItems: "
          f"{data.ItemId.nunique()}\n\tSpan: {data_start.isoformat()} / {data_end.isoformat()}")

    start = datetime.fromtimestamp(data.Time.min(), timezone.utc) + timedelta(days_offset)
    middle = start + timedelta(days_train)
    end = middle + timedelta(days_test)

    # prefilter the timespan
    # make sure a session is not devided over train and test
    session_max_times = data.groupby('SessionId').Time.max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    sub = greater_start[greater_start.isin(lower_end)]
    data_filtered = data[data.SessionId.isin(sub)]

    print(
        f"Slice data set {slice_id}\n\tEvents: {len(data_filtered)}\n\tSessions: {data_filtered.SessionId.nunique()}\n\t"
        f"Items: {data_filtered.ItemId.nunique()}\n\tSpan: {start.date().isoformat()} / {middle.date().isoformat()} / {end.date().isoformat()}")

    smt = data.groupby('SessionId').Time.max().reset_index()['SessionId']

    # split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index

    train = data[data.SessionId.isin(sessions_train)]

    print(f"Train set {slice_id}\n\tEvents: {len(train)}\n\tSessions: {train.SessionId.nunique()}\n\t"
          f"Items: {train.ItemId.nunique()}\n\tSpan: {start.date().isoformat()} / {middle.date().isoformat()}")

    train.to_csv(output_file + '_train.' + str(slice_id) + '.txt', sep='\t', index=False)

    # create test set
    # make sure items which are in train are also in test, cannot predict items which are not in train
    test = data[data.SessionId.isin(sessions_test)]
    test = test[test.ItemId.isin(train.ItemId)]

    # make sure the sessions in the test set have at least 2 unique item views (only in case visit covers 2 days)
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print(f"Test set {slice_id}\n\tEvents: {len(test)}\n\tSessions: {test.SessionId.nunique()}\n\t"
          f"Items: {test.ItemId.nunique()}\n\tSpan: {middle.date().isoformat()} / {end.date().isoformat()} \n\n")

    # test file
    test.to_csv(output_file + '_test.' + str(slice_id) + '.txt', sep='\t', index=False)

    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400 * 7].index  # initial 1 day take 7 days 86400 * 7
    session_valid = session_max_times[
        session_max_times >= tmax - 86400 * 7].index  # initial 1 day take 7 days 86400 * 7
    train_tr = train[train.SessionId.isin(session_train)]

    valid = train[train.SessionId.isin(session_valid)]
    valid = valid[valid.ItemId.isin(train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    print('Train_tr set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
                                                                           train_tr.ItemId.nunique()))

    train_tr.to_csv(output_file + '_train_tr.' + str(slice_id) + '.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                             valid.ItemId.nunique()))

    valid.to_csv(output_file + '_train_valid.' + str(slice_id) + '.txt', sep='\t', index=False)

# -------------------------------------
# MAIN TEST
# --------------------------------------
if __name__ == '__main__':
    preprocess_slices();


