import json
import string as st
import re
import nltk


class CleanData():

    def __init__(self,
                 data_points=10000,
                 file_name="./data/reviews_Electronics_5.json") -> None:

        # essential parameters
        self.data_points = data_points
        self.file_name = file_name
        self.word2index = {}
        self.index2word = {}
        self.vocab = []
        self.word_count = 0
        self.load_data()
        nltk.download('stopwords')

    def load_data(self):

        # read from file
        with open(self.file_name) as f:
            result = f.read().split("\n")

        # convert to dictionary and extract text
        self.corpus = [json.loads(entry)['reviewText']
                       for entry in result[:self.data_points]]

        del result

    def remove_punct(self, text):
        clean_text = ""
        for ch in text:
            if ch not in st.punctuation:
                clean_text += ch
            else:
                clean_text += " "
        return clean_text
        # return ("".join([ch for ch in text if ch not in st.punctuation]))

    def tokenize(self, text):
        return [x.lower() for x in re.split('\s+', text)]

    def remove_small_words(self, text):
        return [x for x in text if len(x) > 2]

    def remove_not_words(self, text):
        return [word for word in text if not any(ch.isdigit() for ch in word)]

    def remove_stopwords(self, text):
        return [word for word in text if word not in nltk.corpus.stopwords.words('english')]

    def clean_corpus(self, corpus, tokeniseData=True):

        # apply the following pipeline to clean
        clean_data = self.remove_punct(corpus)
        clean_data = self.tokenize(clean_data)
        # clean_data = self.remove_small_words(clean_data)
        clean_data = self.remove_not_words(clean_data)
        # clean_data = self.remove_stopwords(clean_data)

        # returns tokens by default
        return clean_data if tokeniseData else " ".join([word for word in clean_data])

    def initialise(self):

        for i, sentence in enumerate(self.corpus):
            self.corpus[i] = self.clean_corpus(sentence)

        for sentence in self.corpus:
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
