import numpy as np
import os
from ratings_to_ewe import get_ewe
import pandas as pd

WORD_FILE = "input_preprocessing/word_lists/55-3.csv"
RATE_FILE = "ratings_preprocessing/output/55_3"
POSITIVE_THRESHOLD = 60


def add_avg_ewe_to_file(ewe, rate_fp, word_fp):
    avg_ewe = ewe.mean()
    par_num, vid_num = os.path.basename(rate_fp).split("_")
    with open("average_ewe.csv", "a+") as f:
        f.write(",".join([str(par_num), str(vid_num), str(avg_ewe), rate_fp, word_fp])+"\n")


def pair_word_to_ewe(rate_file_path, words_file_path):
    ewe, time_stamp = get_ewe(rate_file_path)
    embedding_df = pd.read_pickle(words_file_path)
    add_avg_ewe_to_file(ewe, rate_file_path, words_file_path)

    def get_millisecond(timestamp):
        str_min, str_sec = timestamp.split(":")
        return (int(str_min) * 60 + int(str_sec)) * 1000

    embedding_df.start = embedding_df.start.apply(get_millisecond)
    embedding_df.end = embedding_df.end.apply(get_millisecond)
    middle_time = embedding_df.apply(lambda row: (row['start'] + row['end']) / 2, axis=1)
    embedding_df['rating'] = np.interp(middle_time, time_stamp, ewe)
    embedding_df.to_pickle(words_file_path)
    return embedding_df


