import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from data import CleanData


class CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim=10):

        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lin = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):

        embeds = torch.mean(self.embeddings(inputs), dim=1)
        out = self.lin(embeds)
        out = F.logsigmoid(out)

        return out

    def get_word_emdedding(self, word, word2index):

        word = torch.tensor([word2index[word]])
        return self.embeddings(word).view(1, -1)


class Cbow(CleanData):

    def __init__(self, window=5, features=10, epochs=20) -> None:

        super().__init__()

        self.window = window
        self.features = features
        self.epochs = epochs
        self.data = []

        self.initialise()
        self.context_target()
        print(f'\nVocab size : {self.word_count}\n')
        self.model = CBOW(self.word_count, features)
        self.loss_function = nn.NLLLoss(weight=self.neg_sampling())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def context_target(self):

        for sentence in tqdm(self.corpus):
            for i in range(self.window, len(sentence) - self.window):
                context = sentence[i-self.window: i] + \
                    sentence[i+1: i+self.window+1]
                target = sentence[i]
                self.data.append((context, target))

    def make_context_vector(self, context, word2index, islol=True):

        idxs = [word2index[w] for w in context]
        return torch.tensor(idxs, dtype=torch.long) if islol else idxs

    def neg_sampling(self, num_neg_samples=10):

        freq_distr_norm = F.normalize(
            torch.Tensor(self.word_freq).pow(0.75), dim=0)
        weights = torch.ones(len(self.word_freq))

        for _ in tqdm(range(len(self.word_freq))):
            for _ in range(num_neg_samples):
                neg_ix = torch.multinomial(freq_distr_norm, 1)[0]
                weights[neg_ix] += 1

        return weights

    def train(self):
        losses = []

        for epoch in tqdm(range(self.epochs)):

            total_loss = 0
            batch_size = 50
            total_size = len(self.data)

            for start in tqdm(range(0, total_size, batch_size)):

                batched_data = self.data[start: min(
                    start+batch_size, total_size)]

                context_var = Variable(torch.LongTensor(
                    [self.make_context_vector(datapoint[0], self.word2index, False) for datapoint in batched_data]))

                target_var = Variable(
                    torch.LongTensor([self.word2index[datapoint[1]]
                                     for datapoint in batched_data]))

                self.model.zero_grad()
                log_probs = self.model(context_var)
                loss = self.loss_function(log_probs, target_var)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            losses.append(total_loss)

            print(f'Loss for epoch-{epoch} : {losses[-1]}')
            torch.save(self.model.state_dict(), f'./results/model-{epoch}')
        else:
            torch.save(self.model.state_dict(), f'./results/new_final_model')

        return self.model, losses

    def test(self):

        context = [self.corpus[0][0], self.corpus[0][1],
                   self.corpus[0][2], self.corpus[0][3],
                   self.corpus[0][4], self.corpus[0][6],
                   self.corpus[0][7], self.corpus[0][8],
                   self.corpus[0][9], self.corpus[0][10]]
        target = self.word2index[self.corpus[0][5]]
        context_vector = self.make_context_vector(
            context, self.word2index, False)
        context_var = Variable(torch.LongTensor(context_vector))

        self.model.zero_grad()
        log_probs = self.model(context_var)
        predicted_word = self.index2word[torch.argmax(log_probs[0]).item()]
        print(f'predicted : {predicted_word}')
        print(f'label : {self.index2word[target]}')


if __name__ == '__main__':

    cbow = Cbow()
    cbow.train()
    # cbow.test()
