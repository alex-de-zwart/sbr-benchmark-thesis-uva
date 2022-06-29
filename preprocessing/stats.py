import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# mwd = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
cwd = os.getcwd()
folder = '/data/raw/'

df = pd.read_csv(cwd + folder + 'cleaned_complete_dataset_check.txt', sep='\t')


def get_stats(df):
    """
    :param df: dataframe with SessionId, ItemId, Time
    :return: characteristics of the dataset
    """

    print(f'Events: {df.ItemId.count()}')
    print(f'Users: {df.VisitorId.nunique()}')
    print(f'Sessions: {df.SessionId.nunique()}')
    print(f'Items: {df.ItemId.nunique()}')
    print(f'Sessions per User: {round(df.SessionId.nunique() / df.VisitorId.nunique(), 2)}')
    print(f'Actions per Session: {round(df.ItemId.count() / df.SessionId.nunique(), 2)}')



get_stats(df)



session_lenghts = np.array(df.groupby('SessionId')['ItemId'].size().reset_index(name='size')['size'])
np.percentile(session_lenghts, 99)




