import os

from gensim.models import FastText
import pandas as pd
from tqdm import tqdm


def update_with_ft_vectors(file_path, model):
    try:
        word = None
        word_list_df = pd.read_csv(file_path)
        vectors = []
        for word in tqdm(word_list_df.word):
            word = word.replace("*", "").replace("_", "")
            vec = model.wv.get_vector(word)
            vectors.append(vec)
        word_list_df["fasttext_vector"] = vectors
        word_list_df.to_pickle(file_path)
        good_files.append(file_path)
    except Exception as e:
        print(f"problem with {file_path},{word}")
        print(e)


model = FastText.load_fasttext_format("cc.he.300.bin")
files = os.listdir("word_lists")
good_files = []
for wl_file in tqdm(files):
    file_path = os.path.join("word_lists", wl_file)
    update_with_ft_vectors(file_path, model)

with open("good.csv", "w") as of:
    for fp in good_files:
        of.write(f"{fp}\n")
