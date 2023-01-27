import torch
from torch import nn
import numpy as np
import pandas as pd

"""
Format of input data:
    date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
    2016-07-01 00:00:00,5.827000141143799,2.009000062942505,1.5989999771118164,0.4620000123977661,4.203000068664552,1.3400000333786009,30.5310001373291
- We are trying to predict the Oil temperature OT
- We will place these values straight into a torch tensor
"""
def transform_dataset(input_file_name, output_file_dir, data_title):
    df = pd.read_csv(input_file_name, index_col='date', parse_dates=True)
    lengths = set()

    training_dataset = []
    for i,date,data in enumerate(df.iterrows()):
        if (i > 10):
            break
        transformed_data = transform_data_line(date, data)
        training_dataset.append(transformed_data)



def transform_data_line(date, data):
    date_val = date.timestamp()


def main():
    transform_dataset("./ETT-small/ETTh1.csv", "./data-ett/", "train")
    transform_dataset("./ETT-small/ETTh2.csv", "./data-ett/", "test")

if __name__ == "__main__":
    main()