import torch
from torch import nn
import random
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=4):
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_size = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2
        )
        self.fc = nn.Linear(self.lstm_size, vocab_size)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding_layer = nn.Embedding(self.input_dim, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            n_layers, dropout=dropout)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, input):
        embedding = self.dropout(self.embedding_layer(input))
        output, (state_h, state_c) = self.lstm(embedding)
        return state_h, state_c


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding_layer = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            n_layers, dropout=dropout)
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, input, state_h, state_c):
        input = input.unsqueeze(0)
        embedding = self.dropout(self.embedding_layer(input))
        output, (state_h, state_c) = self.lstm(embedding, (state_h, state_c))
        pred = self.linear(output.squeeze(0))
        return pred, state_h, state_c


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input, ground_truth, force_teaching_ratio=0.5):
        state_h, state_c = self.encoder(input)
        outputs = torch.zeros(
            ground_truth.shape[0], ground_truth.shape[1], self.decoder.output_dim).to(self.device)
        decoder_input = ground_truth[0, :]

        for idx in range(1, ground_truth.shape[0]):
            output, state_h, state_c = self.decoder(
                decoder_input, state_h, state_c)
            outputs[idx] = output
            force = random.random() < force_teaching_ratio
            predicted = output.argmax(1)
            decoder_input = ground_truth[idx] if force else predicted

        return outputs
