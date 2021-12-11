# -*- coding:utf-8 -*-
import re
import os.path as path
from glob import glob
from tqdm import tqdm

READ_MODE = "r"
WRITE_MODE = "w"
APPEND_MODE = "a"
WRD_PAT = re.compile(r"\s*(?P<word>[\w_*,.\"0-9`]+\s?_?)(?P<time>\d\d:\d\d)_?\*?\*?")
NUM_WRD_PAT = re.compile(r"(?P<time2>\d\d:\d\d)\s*(?P<word2>\s\d+\s)")
OUTPUT_DIR = "word_lists"
INPUT_DIR = "transcripts"
titles = ["word", "start", "end"]

LAST_ELEM = -1
TRANSCRIPT_FILE_SUF_LEN = 2
OUTPUT_FILE_TYPE = "csv"


def output_words(transcript_path):
    with open(transcript_path, READ_MODE, encoding="utf-8") as f:
        transcript = f.read()
        words_iter = re.findall(WRD_PAT, transcript)
        for i, match in enumerate(words_iter[:LAST_ELEM]):
            words_iter[i] = list(match) + [words_iter[i + 1][LAST_ELEM]]
        if words_iter:
            words_iter[LAST_ELEM] = list(words_iter[LAST_ELEM]) + [words_iter[LAST_ELEM][LAST_ELEM]]
        else:
            print(f"Error with {transcript_path}!")
            return []

        num_words = re.findall(NUM_WRD_PAT, transcript)
        if num_words:
            for i, match in enumerate(num_words[:LAST_ELEM]):
                num_words[i] = list(match)[::LAST_ELEM] + [words_iter[i + 1][0]]
            num_words[LAST_ELEM] = list(num_words[LAST_ELEM])[::LAST_ELEM] + [num_words[LAST_ELEM][0]]
        words_iter += num_words
    with open(OUTPUT_DIR + path.sep +
              path.basename(transcript_path)[:-TRANSCRIPT_FILE_SUF_LEN]
              + OUTPUT_FILE_TYPE, WRITE_MODE, encoding="utf-8") as o:
        o.write(",".join(titles) + "\n")
        for word in words_iter:
            word = [w.replace(",", "") for w in word]
            o.write(",".join(word) + "\n")


transcript_files = glob(INPUT_DIR + path.sep + "*.md")

for transcript_file in tqdm(transcript_files):
    output_words(transcript_file)
