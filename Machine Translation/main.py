from data import CleanData
from model import Model
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class EuroData(Dataset):
    def __init__(self, data, seq_len=4) -> None:
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.word_sequence = [
            data.word2index[word] for sentence in data.corpus for word in sentence]

    def __len__(self):
        return self.data.word_count - self.seq_len

    def __getitem__(self, i):
        word_indices = torch.tensor(self.word_sequence[i:i+seq_len])
        next_word_indices = torch.tensor(self.word_sequence[i+1:i+seq_len+1])
        return word_indices, next_word_indices


if __name__ == '__main__':

    # TrainData = CleanData("./data/europarl-corpus/train.europarl")
    TrainData = CleanData("./data/news-crawl-corpus/train.news", 4, 'french')
    TrainData.initialise()
    # for sent in TrainData.corpus:
    #     print(f"\n{sent}")
    print(f"\nNumber of Sentences: {len(TrainData.corpus)}")
    print(f"Vocab Size: {TrainData.word_count}")
    print(f"Max Sentence Length: {TrainData.max_sent_size}")
    print(f"Min Sentence Length: {TrainData.min_sent_size}\n")

    seq_len = 4
    epochs = 20

    train_data = EuroData(TrainData, 4)
    train_loader = DataLoader(train_data, batch_size=64, num_workers=4)

    # embedding dimension -> vector size of each sentence
    # hidden dimension -> hidden dimension
    # number of layers -> no.of words sent and the no.of LSTM blocks
    # output size -> output vector dimension
    model = Model(vocab_size=TrainData.word_count,
                  embedding_dim=256,
                  hidden_dim=64,
                  num_layers=4)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        state_h, state_c = model.init_state(seq_len)
        # print(state_h.shape, state_c.shape)

        model.train()
        for X, y in tqdm(train_loader):

            optimizer.zero_grad()

            # Forward pass
            y_pred, (state_h, state_c) = model(
                X, (state_h, state_c))

            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            # Backward and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch-{epoch} Loss: {loss.item()}")
        torch.save(model, 'models/lm1_french.pth')
