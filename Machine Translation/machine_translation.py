import random
from data import CleanData
from tqdm import tqdm
import numpy as np
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data import best_word, sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
seq_len = 0

EnglishData = CleanData("./data/ted-talks-corpus/train.en", seq_len, 'english')
FrenchData = CleanData("./data/ted-talks-corpus/train.fr", seq_len, 'french')


class MyData(Dataset):
    def __init__(self, eng_data, fr_data) -> None:
        super().__init__()
        self.data = list(zip(eng_data, fr_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        eng_sent = []
        for word in self.data[i][0]:
            if word not in EnglishData.vocab:
                word = '<OOV>'
            eng_sent.append(EnglishData.word2index[word])

        fr_sent = []
        for word in self.data[i][1]:
            if word not in FrenchData.vocab:
                word = '<OOV>'
            fr_sent.append(FrenchData.word2index[word])

        return (np.array(eng_sent), np.array(fr_sent))


def collate(data):
    X = [x[0] for x in data]
    Y = [y[1] for y in data]

    x_len = max([len(x) for x in X])
    y_len = max([len(y) for y in Y])

    padded_x = np.zeros((batch_size, x_len))
    padded_y = np.zeros((batch_size, y_len))

    for idx, (x, y) in enumerate(zip(X, Y)):
        padded_x[idx] = np.pad(x, (0, x_len - len(x)))
        padded_y[idx] = np.pad(y, (0, y_len - len(y)))

    return (
        torch.tensor(padded_x, dtype=torch.long).t(),
        torch.tensor(padded_y, dtype=torch.long).t())


def translate(model, text):
    model.eval()
    with torch.no_grad():
        tokens = EnglishData.clean_corpus(text)
        tokens = ['<SOS>'] + tokens + ['<EOS>']
        src_indexes = []
        for token in tokens:
            if token not in EnglishData.vocab:
                token = '<OOV>'
            src_indexes.append(EnglishData.word2index[token])
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        src_tensor = src_tensor.reshape(-1, 1)

        output = model(src_tensor, src_tensor)
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        indices = torch.argmax(output, dim=1).tolist()
        # indices = best_word(len(FrenchData.index2word), output.shape[0])
        return ' '.join([FrenchData.index2word[x] for x in indices])


def calculate_bleu_score(source, target):
    translated = translate(model, ' '.join(
        [EnglishData.index2word[x.item()] for x in source]))[1:]
    score = sentence_bleu(target, translated)
    return score, translated


if __name__ == '__main__':

    EnglishData.initialise()
    FrenchData.initialise()

    EnglishDataTest = CleanData(
        "./data/ted-talks-corpus/test.en", seq_len, 'english')
    FrenchDataTest = CleanData(
        "./data/ted-talks-corpus/test.fr", seq_len, 'french')
    EnglishDataTest.initialise()
    FrenchDataTest.initialise()

    test_data = MyData(EnglishDataTest.corpus, FrenchDataTest.corpus)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             num_workers=4,
                             collate_fn=collate,
                             drop_last=True)

    # Define Models
    model_path = sys.argv[1]
    model = torch.load(model_path)

    words = input('Enter sentence to be translated').split()
    print(translate(model, words))
    r = input('Proceed? [y]/n')
    if r != 'y':
        exit()

    total_score = 0
    candidates, references = list(), list()
    for source, target in test_loader:
        # print(' '.join([EnglishData.index2word[x.item()] for x in source]))
        # print(' '.join([FrenchData.index2word[x.item()] for x in target]))
        # print(translate(model, ' '.join(
        #     [EnglishData.index2word[x.item()] for x in source[1:-1]])))
        # print()
        score, translated = calculate_bleu_score(source, target)
        candidates.append(translated[1:])
        references.append(target)
        total_score += score
        print(' '.join([FrenchData.index2word[x.item()]
              for x in target[1:-1]]), score)

    # print(f'Total Score: {total_score}')
    # print('Corpus Score', corpus_bleu(references, candidates))
    print('Corpus Score', total_score/len(test_loader))
