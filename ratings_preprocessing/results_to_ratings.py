import pandas as pd
from os import path

from glob import glob
import json

DATA_DIR = "data"
OUTPUT_DIR = "output"
CSV_PATTERN = "*.csv"

NAME_SUFFIX = "name"
RATING_SUFFIX = "rating"
TIME_SUFFIX = "time"
S = "s"

FIRST_ROW = 2


def load_results_file():
    csv_files = glob(path.join(DATA_DIR, CSV_PATTERN))
    assert len(csv_files) == 1, "Only one CSV file should be in the data dir"
    results_df = pd.read_csv(csv_files[0])

    return results_df.iloc[FIRST_ROW:, :]


NAME_COLUMNS = [f"vid_{i}_{NAME_SUFFIX}" for i in range(1, 5)]
TIME_COLUMNS = [f"vid_{i}_{TIME_SUFFIX}" for i in range(1, 5)]
RATING_COLUMNS = [f"vid_{i}_{RATING_SUFFIX}" for i in range(1, 5)]


def get_data_from_row(row):
    data_dict = {}
    for i, name_col in enumerate(NAME_COLUMNS):
        data_dict[i] = {NAME_SUFFIX: row[name_col]}
    for i, rating_col in enumerate(RATING_COLUMNS):
        data_dict[i][RATING_SUFFIX] = row[rating_col]
    for i, time_col in enumerate(TIME_COLUMNS):
        data_dict[i][TIME_SUFFIX] = row[time_col]
    s = row[S]
    if pd.isna(s):
        s = row["ResponseId"]
    data_dict[S] = s

    return data_dict


df = load_results_file()
results = {}

for i, r in df.iterrows():
    results[i] = get_data_from_row(r)

videos_data = {}
videos_meta_data = {}


def add_video_data(u_data, u_s):
    vid_name = u_data["name"]
    assert not pd.isna(vid_name)
    user_data = {"participant": u_s, "rating": u_data["rating"].split(",") if not pd.isna(u_data["rating"]) else "",
                 "time": u_data["time"].split(",") if not pd.isna(u_data["time"]) else ""}

    if not videos_data.get(vid_name):
        videos_data[vid_name] = [user_data]
        videos_meta_data[vid_name] = {"name": vid_name, "valid": 0, "all": 1}
    else:
        videos_data[vid_name].append(user_data)
        videos_meta_data[vid_name]["all"] += 1

    valid_data = not pd.isna(u_data["rating"]) and not pd.isna(u_data["time"])
    if valid_data:
        videos_meta_data[vid_name]["valid"] += 1


for i, data in results.items():
    s = data["s"]
    assert not pd.isna(s)
    for j, vid_data in data.items():
        if j != "s":
            add_video_data(vid_data, s)

for vid, data in videos_data.items():
    pd.DataFrame(data).to_csv(f"{OUTPUT_DIR}{path.sep}{vid}", index=False)
pd.DataFrame(videos_meta_data.values()).to_csv("videos_meta.csv",index=False)