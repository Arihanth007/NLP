from data import CleanData
from model import Encoder, Decoder, Seq2Seq
from tqdm import tqdm
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
seq_len = 4
epochs = 10

EnglishData = CleanData("./data/ted-talks-corpus/train.en", seq_len, 'english')
FrenchData = CleanData("./data/ted-talks-corpus/train.fr", seq_len, 'french')


class MyData(Dataset):
    def __init__(self, eng_data, fr_data) -> None:
        super().__init__()
        self.data = list(zip(eng_data, fr_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        eng_sent = np.array([EnglishData.word2index[word]
                            for word in self.data[i][0]])
        fr_sent = np.array([FrenchData.word2index[word]
                           for word in self.data[i][1]])
        return (eng_sent, fr_sent)


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


def train(model, optimizer, criterion, dataloader):

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, y)
            pred = pred[1:].reshape(-1, pred.shape[-1])
            y = y[1:].reshape(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print({'epoch': epoch, 'loss': epoch_loss/len(dataloader)})
        torch.save(model, './models/translate.pth')


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
        return ' '.join([FrenchData.index2word[x] for x in indices])


if __name__ == '__main__':
    EnglishData.initialise()
    FrenchData.initialise()

    enc = Encoder(len(EnglishData.vocab), 256, 128, seq_len, 0.2)
    dec = Decoder(len(FrenchData.vocab), 256, 128, seq_len, 0.2)
    model = Seq2Seq(enc, dec, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_data = MyData(EnglishData.corpus, FrenchData.corpus)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              num_workers=4,
                              collate_fn=collate,
                              drop_last=True)

    train(model, optimizer, criterion, train_loader)
    print(translate(model, "and we're going to tell you some stories from the sea"))
