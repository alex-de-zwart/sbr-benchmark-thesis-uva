import os
import pandas as pd


for root, dirs, files in os.walk('data/prepared/vsknn', topdown=False):
    for name in files:
        path = os.path.join(root, name)
        df = pd.read_csv(path, sep='\t', encoding='utf-8')
        print(f'Number of items {df.ItemId.nunique()} in file: {name}')

