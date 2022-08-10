import json
import string as st
import re
import nltk
import numpy as np
from scipy.spatial import distance
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from data import CleanData


class WordEmbedding(CleanData):

    def __init__(self, window=3, features=50) -> None:

        super().__init__()

        self.window = window
        self.features = features
        self.initialise()

    def cooccurence_matrix(self):

        self.sparse_matrix = lil_matrix(
            (self.word_count, self.word_count), dtype=np.float32)

        for sentence in tqdm(self.corpus):
            for i in range(0, len(sentence) - self.window + 1):
                for j in range(i + 1, i + self.window + 1):
                    if j < len(sentence):
                        i_index = self.word2index[sentence[i]]
                        j_index = self.word2index[sentence[j]]
                        self.sparse_matrix[i_index, j_index] += 1
                        self.sparse_matrix[j_index, i_index] += 1

        return self.sparse_matrix

    def svd(self, matrix):

        right, sigma, left_t = svds(matrix, self.features)
        self.embeddings = right
        return right

    def save(self, file_name='./results/q1_embeddings.json'):

        word_embeddings = {}
        for i in range(self.word_count):
            word_embeddings[self.index2word[i]] = self.embeddings[i]

        with open(file_name, 'w+', encoding='utf-8') as f:
            json.dump(word_embeddings, f, ensure_ascii=False,
                      indent=4, cls=NumpyEncoder)

    def analysis(self, num_words=10):

        # test_on_words = ['asking', 'father']
        test_on_words = ['sad', 'eat', 'farm', 'valentine', 'camera']
        test_on_words_indices = [self.word2index[word]
                                 for word in test_on_words]

        for word_index in test_on_words_indices:

            cosine_dists = [(1-distance.cosine(
                self.embeddings[word_index], self.embeddings[i]), i) for i in range(self.word_count)]

            cosine_dists.sort(key=lambda x: x[0], reverse=True)

            print(f'\n{self.index2word[word_index]}')
            for closest_word in cosine_dists[:11]:
                print(
                    f'{self.index2word[closest_word[1]]} : {closest_word[0]}')

            X = [self.embeddings[emb[1]] for emb in cosine_dists[0:11]]
            X = np.array(X)
            X_embedded = TSNE(n_components=2, learning_rate='auto',
                              init='random').fit_transform(X)

            # sns.set(rc={'figure.figsize': (12, 12)})
            # sns.scatterplot(
            #     x=X_embedded[:, 0], y=X_embedded[:, 1], label=f'{self.index2word[word_index]}')
            # plt.savefig(f'./img/{self.index2word[word_index]}.png')

        # plt.legend()
        # plt.savefig(f'./img/q1.png')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def time_task(task):

    start = time.time()
    task()  # perform task
    end = time.time()
    print(f"\n\nTime Elapsed: {end - start}\n")


if __name__ == '__main__':

    def task1():

        word_embeddings = WordEmbedding()
        sparse_matrix = word_embeddings.cooccurence_matrix()
        embeddings = word_embeddings.svd(sparse_matrix)
        word_embeddings.save()
        word_embeddings.analysis()

        print(f"\n{len(embeddings)}, {word_embeddings.word_count}\n")

    time_task(task1)
