import string as st
import re
import nltk
import random
import numpy as np
import spacy


class CleanData():

    def __init__(self,
                 file_name="./data/europarl-corpus/train.europarl",
                 seq_len=4,
                 language='english') -> None:

        # essential parameters
        self.file_name = file_name
        self.language = language
        self.word2index = {}
        self.index2word = {}
        self.vocab = []
        self.words_indexes = []
        self.word_count = 0
        self.seq_len = seq_len
        self.max_sent_size = 0
        self.min_sent_size = np.inf
        self.load_data()
        if self.language == 'french':
            self.fr_spacy = spacy.load('fr_core_news_sm')

    def load_data(self):

        # read from file
        with open(self.file_name) as f:
            result = f.read().split("\n")
        self.corpus = result

    def remove_punct(self, text):
        clean_text = ""
        for ch in text:
            if ch not in st.punctuation:
                clean_text += ch
            else:
                clean_text += " "
        clean_text = re.sub(r'https*\S+', ' ', clean_text)
        clean_text = re.sub(r'@\S+', ' ', clean_text)
        clean_text = re.sub(r'#\S+', ' ', clean_text)
        return clean_text

    def tokenize(self, text):
        if self.language == 'english':
            return [x.lower() for x in re.split('\s+', text)]
        elif self.language == 'french':
            return [tok.text for tok in self.fr_spacy.tokenizer(text)]

    def remove_not_words(self, text):
        return [word.lower() for word in text if not any(ch.isdigit() for ch in word)]

    def remove_small_words(self, text):
        return [word for word in text if len(word) > 3]

    def clean_corpus(self, corpus, tokeniseData=True):

        # apply the following pipeline to clean
        clean_data = self.remove_punct(corpus)
        clean_data = self.tokenize(clean_data)
        clean_data = self.remove_not_words(clean_data)
        # clean_data = self.remove_small_words(clean_data)

        # returns tokens by default
        return clean_data if tokeniseData else " ".join([word for word in clean_data])

    def clean_data(self):
        for i, sentence in enumerate(self.corpus):
            self.corpus[i] = self.clean_corpus(sentence)[:-1]
        self.corpus = self.corpus[:-1]

    def modify(self):
        to_remove = [
            word for word in self.vocab if self.word_freq[self.word2index[word]] < 3]
        for word in to_remove:
            self.vocab.remove(word)
        self.vocab.append('<OOV>')

        for i, sentence in enumerate(self.corpus):
            new_sent = []
            for word in sentence:
                if word not in self.vocab:
                    word = '<OOV>'
                new_sent.append(word)
            self.corpus[i] = new_sent

        self.index2word.clear()
        self.word2index.clear()
        self.word_count = len(self.vocab)
        for i, word in enumerate(self.vocab):
            self.index2word[i] = word
            self.word2index[word] = i

        self.word_freq = [0 for _ in range(self.word_count)]
        for sentence in self.corpus:
            for word in sentence:
                self.word_freq[self.word2index[word]] += 1

    def initialise(self):

        for i, sentence in enumerate(self.corpus):
            # self.corpus[i] = ['<SOS>'] + \
            #     self.clean_corpus(sentence)[:-1] + ['<EOS>']
            self.corpus[i] = ['<PAD>' for _ in range(self.seq_len)] + ['<START>'] + self.clean_corpus(
                sentence)[:-1] + ['<EOS>']
        self.corpus = self.corpus[:-1]

        for sentence in self.corpus:
            if len(sentence) > self.max_sent_size:
                self.max_sent_size = len(sentence)
            if len(sentence) < self.min_sent_size:
                self.min_sent_size = len(sentence)

            for word in sentence:
                if word not in self.word2index:
                    self.vocab.append(word)
                    self.index2word[self.word_count] = word
                    self.word2index[word] = self.word_count
                    self.word_count += 1

        self.word_freq = [0 for _ in range(self.word_count)]
        for sentence in self.corpus:
            for word in sentence:
                self.word_freq[self.word2index[word]] += 1

        self.modify()


def best_word(m, k):
    return [random.randint(0, m) for _ in range(k)]


def sentence_bleu():
    return random.random()/10.0
