import os.path

import pandas as pd
import numpy as np

INTERP_RATING = 'interp_rating'

RATE_TIME_STEP = 500



def get_np_arrays(col):
    def str_to_array(str_array):
        return np.array(str_array.replace("[", "").replace("]", "").replace(" ", "").replace("'", "").split(','),
                        dtype=int)

    return col[col.notnull()].apply(str_to_array)


def get_rating_df(file_path):
    df = pd.read_csv(file_path)
    df = pd.concat([df.participant[df.time.notnull()], get_np_arrays(df.time), get_np_arrays(df.rating)], axis=1)
    df = df[df['rating'].apply(lambda x: np.unique(x).size > 1)]
    assert df.size > 1
    max_vector_len = df.time.apply(len).max()
    assert max_vector_len > 0, "prob"
    range_to_interp = np.arange(0, max_vector_len * RATE_TIME_STEP, RATE_TIME_STEP)
    df[INTERP_RATING] = df.apply(lambda row: np.interp(range_to_interp, row['time'], row['rating']), axis=1)
    return df, range_to_interp


def get_ewe(file_path):
    rating_df, time_stamps = get_rating_df(file_path)
    rate_mat = np.stack(rating_df[INTERP_RATING])
    assert rate_mat.shape[0] > 1, "We must have more than one rating per video"
    row_indices = np.arange(rate_mat.shape[0])

    with open("rating_num_log.csv", "a") as f:
        f.write("{},{}\n".format(os.path.basename(file_path), rate_mat.shape[0]))

    def correlate_with_others(row_ind):
        return np.corrcoef(rate_mat[row_ind, :], rate_mat[row_indices != row_ind, :].mean(axis=0))[0, 1]

    weights = np.array([correlate_with_others(ind) for ind in row_indices])
    return (rate_mat * np.array(weights)[:, None]).sum(axis=0) / sum(weights), time_stamps
