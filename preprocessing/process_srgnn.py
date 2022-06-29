DATA_PATH = '../data/prepared/vsknn/'
RESULT_PATH = '../data/prepared/srgnn/'
MODEL_FILES = ['item_views_train', 'item_views_test', 'item_views_train_tr', 'item_views_train_valid']

import pandas as pd
import datetime
import operator
from datetime import datetime, timezone, timedelta
import numpy as np
import pickle
import os
from itertools import chain

def load_data(file_path: str) -> pd.DataFrame:
    """

    :param file_path: location of the import dataset
    :return: pandas dataframe
    """

    print('start load_data..')
    # load csv
    data = pd.read_csv(file_path, sep='\t', header=None, skiprows=1, usecols=[0, 1, 2])
    # specify header names
    data.columns = ['session_id', 'item', 'time_stamp']

    # output
    data_start = datetime.fromtimestamp(data.time_stamp.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.time_stamp.max(), timezone.utc)

    print(
        f"Loaded data set\n\tEvents: {len(data)}\n\tSessions: {data.session_id.nunique()}\n\tItems: {data.item.nunique()}"
        f"\n\tSpan: {data_start.date().isoformat()} / {data_end.date().isoformat()}\n\n")

    return data

def format_data(df) -> dict:
    """

    :param file_path: path of raw dataset
    :return: session_clicks dict with session_id and list of items belonging to that session
    sess_date: dict with session_id and epoch timestamp
    """

    sess_clicks = {}
    sess_date = {}
    ctr = 0
    current_id = -1
    current_date = None

    for _, row in df.iterrows():
        session_id = row["session_id"]
        if current_date and not current_id == session_id:
            date = ''
            date = current_date
            sess_date[current_id] = date
        current_id = session_id
        item = row['item'], row['time_stamp']
        current_date = row['time_stamp']

        if session_id in sess_clicks:
            sess_clicks[session_id] += [item]
        else:
            sess_clicks[session_id] = [item]
        ctr += 1
    date = current_date
    for i in list(sess_clicks):
        sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
        sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[current_id] = date

    print("-- Reading data @ %ss" % datetime.now())

    return sess_clicks, sess_date

# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra(tra_sess, train_sess_clicks):
    item_dict = {}
    reversed_item_dict = {}
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = train_sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                reversed_item_dict[item_ctr] = i
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]

    return train_ids, train_dates, train_seqs, item_dict, reversed_item_dict


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes(tes_sess, test_sess_clicks, item_dict):
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = test_sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]

    return test_ids, test_dates, test_seqs


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[i:]
            labs += [tar]
            out_seqs += [seq[:i]]
            out_dates += [date]
            ids += [id]

    return out_seqs, out_dates, labs, ids


if __name__ == '__main__':

    for i in range(3):
        train = load_data(DATA_PATH + 'item_views_train.' + str(i) + '.txt')
        test = load_data(DATA_PATH + 'item_views_test.' + str(i) + '.txt')

        train_sess_clicks, train_session_dates = format_data(train)
        test_sess_clicks, test_session_dates = format_data(test)

        train_session_dates = list(train_session_dates.items())
        test_session_dates = list(test_session_dates.items())

        tra_ids, tra_dates, tra_seqs, item_dict, reversed_item_dict = obtian_tra(train_session_dates, train_sess_clicks)
        tes_ids, tes_dates, tes_seqs = obtian_tes(test_session_dates, test_sess_clicks, item_dict)

        tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
        te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
        print(f'Items which are in test but not in train item_views_train {str(i)}: {set(te_ids)-set(tr_ids)}')
        tra = (tr_seqs, tr_labs)
        tes = (te_seqs, te_labs)

        # write files
        if not os.path.exists('../data/prepared/srgnn'):
            os.makedirs('../data/prepared/srgnn')
        pickle.dump(tra, open(RESULT_PATH + 'item_views_train.' + str(i) + '.txt', 'wb'))
        pickle.dump(tes, open(RESULT_PATH + 'item_views_test.' + str(i) + '.txt', 'wb'))
        pickle.dump(item_dict, open(RESULT_PATH + 'item_views_train.' + str(i) + '.reversed_item_dict' + '.pkl', 'wb'))

    for i in range(3):
        train = load_data(DATA_PATH + 'item_views_train_tr.' + str(i) + '.txt')
        test = load_data(DATA_PATH + 'item_views_train_valid.' + str(i) + '.txt')

        train_sess_clicks, train_session_dates = format_data(train)
        test_sess_clicks, test_session_dates = format_data(test)

        train_session_dates = list(train_session_dates.items())
        test_session_dates = list(test_session_dates.items())

        tra_ids, tra_dates, tra_seqs, item_dict, reversed_item_dict = obtian_tra(train_session_dates, train_sess_clicks)
        tes_ids, tes_dates, tes_seqs = obtian_tes(test_session_dates, test_sess_clicks, item_dict)

        tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
        te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
        print(f'Items which are in test but not in train item_views_train_tr {str(i)}: {set(te_ids)-set(tr_ids)}')
        tra = (tr_seqs, tr_labs)
        tes = (te_seqs, te_labs)

        # write files
        if not os.path.exists('../data/prepared/srgnn'):
            os.makedirs('../data/prepared/srgnn')
        pickle.dump(tra, open(RESULT_PATH + 'item_views_train_tr.' + str(i) + '.txt', 'wb'))
        pickle.dump(tra_seqs, open(RESULT_PATH + 'item_views_tra_seqs.' + str(i) + '.txt', 'wb'))
        pickle.dump(tes, open(RESULT_PATH + 'item_views_train_valid.' + str(i) + '.txt', 'wb'))
        pickle.dump(item_dict, open(RESULT_PATH + 'item_views_train_tr.' + str(i) + '.reversed_item_dict' + '.pkl', 'wb'))
