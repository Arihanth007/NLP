from data import CleanData
import torch
import numpy as np
from tqdm import tqdm
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seq_len = 4


# def predict(data, model, words, next_words=10):
#     model.eval()

#     state_h, state_c = model.init_state(len(words))

#     with torch.no_grad():
#         for i in range(0, next_words):

#             x = torch.tensor([[data.word2index[w] for w in words[i:]]])
#             y_pred, (state_h, state_c) = model(x, (state_h, state_c))

#             last_word_logits = y_pred[0][-1]
#             p = torch.nn.functional.softmax(
#                 last_word_logits, dim=0).detach().numpy()
#             word_index = np.random.choice(len(last_word_logits), p=p)
#             words.append(data.index2word[word_index])

#     return words


def perplexity(train_data, test_data, model, seq_len=4):

    model.eval()
    perplex = 0

    with torch.no_grad():
        # for sentence in tqdm(test_data.corpus):
        # for sentence in test_data.corpus:
        words = ['<PAD>' for _ in range(seq_len)] + ['<START>']
        for word in sentence:
            if word not in train_data.vocab:
                word = '<OOV>'
            words.append(word)
        words.append('<EOS>')

        state_h, state_c = model.init_state(seq_len)
        pred_words = []
        perp = 1.0
        orig = [w for w in words]

        for i in range(len(words)-seq_len-1):

            x = torch.tensor([[train_data.word2index[w]
                               for w in words[i:i+seq_len]]])
            y = train_data.word2index[words[i+seq_len+1]]
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))

            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(
                last_word_logits, dim=0)

            word_index = torch.argmax(p).item()
            perp *= ((1/p[y])**(1/(len(words))))
            pred_words.append(train_data.index2word[word_index])

        perplex += perp
        print(f'{perp} {" ".join(orig[1+seq_len:-1])}\n')

    print(f'Average perplexity = {perplex/len(test_data.corpus)}')


if __name__ == '__main__':

    TrainData = CleanData("./data/europarl-corpus/train.europarl")
    TestData = CleanData("./data/europarl-corpus/test.europarl")
    # TrainData = CleanData("./data/news-crawl-corpus/train.news", 4, 'french')
    # TestData = CleanData("./data/news-crawl-corpus/test.news", 4, 'french')
    TrainData.initialise()
    TestData.initialise()

    # model = torch.load('models/lm1_english.pth')
    # model = torch.load('models/lm1_french.pth')
    model_path = sys.argv[1]
    model = torch.load(model_path)

    sentence = input('Enter a sentence: ')

    pred_words = perplexity(TrainData, sentence.split(), model)
    # pred_words = perplexity(TrainData, TestData, model)
    # pred_words = perplexity(TrainData, TrainData, model)
