import re
import sys
import numpy as np
import matplotlib.pyplot as plt


class Tokeniser():
    def __init__(self) -> None:
        self.hastags = r"(#+\w+)"
        self.mentions = r"(@+\w+)"
        self.urls = r"([\w+]+\:\/\/)?([\w\d-]+\.)*[\w-]+[\.\:]\w+([\/\?\=\&\#\.]?[\w-]+)*\/?"
        self.punctuations = r"[\!\"\&\:\;\@\`\.\-\,\?\=\'\[\]\{\}\(\)\* ]"
        self.special_characters = ['!', '"', '#', '$', '%', '&',
                                   '(', ')', '*', '+', '/', ':',
                                   ';', '<', '=', '>', '@', '[',
                                   '\\', ']', '^', '`', '{', '|',
                                   '}', '~', ',']
        self.sentences = []
        self.low_freq_threshold = 3
        with open(sys.argv[3], 'r') as file:
            # sentences = file.read()
            while (line := file.readline().rstrip()):
                self.sentences.append(line)

    # returns a string
    def clean_data(self):
        for i in range(len(self.sentences)):
            # replace hastags, mentions, URLS
            self.sentences[i] = re.sub(
                self.hastags, r"<HASHTAG>", self.sentences[i])
            self.sentences[i] = re.sub(
                self.mentions, r"<MENTIONS>", self.sentences[i])
            self.sentences[i] = re.sub(
                self.urls, r"<URL>", self.sentences[i])

            # replace recurring occurances of special characters
            self.sentences[i] = re.sub(
                r"(\!)\1{2,}", r"!", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\")\1{2,}", r"\"", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\#)\1{2,}", r"#", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\$)\1{2,}", r"$", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\%)\1{2,}", r"%", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\&)\1{2,}", r"&", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\()\1{2,}", r"(", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\))\1{2,}", r")", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\*)\1{2,}", r"*", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\:)\1{2,}", r":", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\;)\1{2,}", r";", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\?)\1{2,}", r"?", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\<)\1{2,}", r"<", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\>)\1{2,}", r">", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\')\1{2,}", r"'", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\[)\1{2,}", r"[", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\])\1{2,}", r"]", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\`)\1{2,}", r"`", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\~)\1{2,}", r"~", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\-)\1{2,}", r"-", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\_)\1{2,}", r"_", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\{)\1{2,}", r"{", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\})\1{2,}", r"}", self.sentences[i])
            self.sentences[i] = re.sub(
                r"(\+)\1{2,}", r"+", self.sentences[i])

        return self.sentences

    # returns a list of words
    def add_buffer(self, n, sentence):
        # pad the sentence so a n-gram
        # can be parsed from the begining
        sentence = sentence.lower()
        sentence = re.split(self.punctuations, sentence)

        buffered_sentence = []
        for _ in range(n-1):
            buffered_sentence.append("<START>")
        for word in sentence:
            buffered_sentence.append(word)
        for _ in range(n-1):
            buffered_sentence.append("<END>")

        # remove empty strings
        new_sentence = [word for word in buffered_sentence if len(word) > 1]
        return new_sentence

     # returns a list of words
    def dense_clean(self, sentences):
        # find wordcount and replace rare ones
        word_freq = {}
        total_words = 0
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                # get rid of filler words and typos
                if word not in self.special_characters and (len(word) < 2 or (any(c.isdigit() for c in word))):
                    sentences[i][j] = '<DEFAULT>'
                    word = '<DEFAULT>'
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
                total_words += 1

        # get rid of rarely occuring words
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                if word_freq[word] < self.low_freq_threshold:
                    sentences[i][j] = '<DEFAULT>'

        # recompute the frequency distribution
        word_freq = {}
        total_words = 0
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
                total_words += 1

        return sentences, word_freq, total_words


class Functions():
    def __init__(self) -> None:
        pass

    # returns dictionary of every n-words
    # and the word that comes right after
    def create_ngram_dictionary(self, n, sentences):
        if n < 0:
            print('Invalid n-gram count')
            return
        if n == 0:
            return {}

        ngrams = {}
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                if j < n-1:
                    continue

                # key contains n-1 words
                so_far_key = " ".join(sentence[j-n+1:j])

                # create a new dictionary entry
                if so_far_key not in ngrams.keys():
                    ngrams[so_far_key] = {}

                # increment known values
                if word in ngrams[so_far_key].keys():
                    ngrams[so_far_key][word] += 1
                # intialise the new word associated
                else:
                    ngrams[so_far_key][word] = 1

        return ngrams

    # returns the probability of the 1-gram
    def simple_smoothing(self, gram_idx, grams, idx, sentence, word_freq):
        p = None
        # return once all grams are visited
        if gram_idx >= len(grams):
            if p is None:
                # done to return a very small probability
                all_words = len(word_freq)  # all unique words
                total_words = sum(word_freq.values())  # occurence of all words
                p = all_words/total_words
            return p

        # visiting a n-gram
        n = len(grams) - gram_idx
        ngram = grams[gram_idx]
        words_so_far = " ".join(sentence[idx-n+1:idx])
        word = sentence[idx]

        try:
            w = len(ngram[words_so_far])  # num words under ngram
            n_w = sum(ngram[words_so_far].values())  # word count under ngram
            wc_w = ngram[words_so_far][word]  # word count of word under ngram

            p = wc_w / n_w
        except:
            p = self.simple_smoothing(
                gram_idx+1, grams, idx, sentence, word_freq)

        return p

    # returns the probability of a word
    # by using information of 1-gram to n-gram
    def witten_bell(self, gram_idx, grams, idx, sentence, word_freq):
        p = None
        # return once all grams are visited
        # get the 1-gram probability
        if gram_idx >= len(grams)-1:
            return self.simple_smoothing(gram_idx, grams, idx, sentence, word_freq)

        # visiting a n-gram
        n = len(grams) - gram_idx
        ngram = grams[gram_idx]
        words_so_far = " ".join(sentence[idx-n+1:idx])
        word = sentence[idx]

        try:
            w = len(ngram[words_so_far])  # num words under ngram
            n_w = sum(ngram[words_so_far].values())  # word count under ngram
            wc_w = ngram[words_so_far][word]  # word count of word under ngram

            p = (n_w+w)**-1 * (wc_w + w * self.witten_bell(gram_idx +
                                                           1, grams, idx, sentence, word_freq))
        except:
            all_words = len(word_freq)  # all unique words
            total_words = sum(word_freq.values())  # occurence of all words

            p = self.witten_bell(gram_idx + 1, grams, idx, sentence, word_freq)
            if p:
                p *= (0.75*all_words/total_words)

        return p

    # returns the probability of the word
    # by using information of 1-gram to n-gram
    def kneser_ney(self, gram_idx, grams, idx, sentence, word_freq):
        # return once all grams are visited
        if gram_idx >= len(grams)-1:
            return self.simple_smoothing(gram_idx, grams, idx, sentence, word_freq)

        # visiting a n-gram
        n = len(grams) - gram_idx
        ngram = grams[gram_idx]
        words_so_far = " ".join(sentence[idx-n+1:idx])
        word = sentence[idx]

        first_term = 0
        try:
            w = len(ngram[words_so_far])  # num words under ngram
            n_w = sum(ngram[words_so_far].values())  # word count under ngram
            wc_w = ngram[words_so_far][word]  # word count of word under ngram

            first_term = max((wc_w-0.75), 0)/n_w

        except:
            pass

        # prior n-gram probability
        try:
            w = len(ngram[words_so_far])  # num words under ngram
            n_w = sum(ngram[words_so_far].values())  # word count under ngram
            l = w/n_w
        except:
            all_words = len(word_freq)  # all unique words
            total_words = sum(word_freq.values())  # occurence of all words
            l = all_words/total_words

        # caclulate the probability of lower n-grams
        pcont = None
        try:
            pcont = 0.75 * l * \
                self.kneser_ney(gram_idx+1, grams, idx, sentence, word_freq)
        except:
            return None

        return first_term + pcont

    # returns perplexity
    def perplexity(self, n, grams, sentence, word_freq, smoothing):
        P = 0
        for j, word in enumerate(sentence):
            if j < n-1:
                continue
            p = smoothing(0, grams, j, sentence, word_freq)
            if p == None:  # Word is not valid
                return
            else:
                P += np.log(p)
        return np.exp(-P/(len(sentence)-n+1))

    def run(self, type_data, data, n, smoothing_func):
        # finding the perplexity
        max_perp = -float("inf")
        min_perp = float("inf")
        total_perp = 0
        counter = 0
        file1 = open(
            f"./results/{sys.argv[3][5:]}-{smoothing_func.__name__}-{type_data}.txt", "a")
        for sentence in data:
            p = self.perplexity(n, grams, sentence, word_freq, smoothing_func)
            if p:
                counter += 1
                total_perp += p
                max_perp = max(max_perp, p)
                min_perp = min(min_perp, p)
                # file1.write(f"{sentence}    {(min_perp+max_perp)/counter}\n")
        file1.close()

        print(f"On {type_data} Set:")
        print(f"Valid Sentences:{counter}/{len(data)}")
        # print(f"Max Perplexity:{max_perp}")
        # print(f"Min Perplexity:{min_perp}")
        print(f"Avg Perplexity:{(min_perp+max_perp)/counter}\n")
        # print(f"Total Perplexity:{total_perp}\n")

        return [counter, max_perp, min_perp, (min_perp+max_perp)/counter]

    def plot_graph(self, info, color, lbl):
        c, max_p, min_p, avg_p = info
        plt.bar(np.log(avg_p), c, color=color, width=0.5, label=lbl)
        plt.annotate("Avg", (np.log(avg_p), c))
        plt.bar(np.log(max_p), c, color=color, width=0.5)
        plt.annotate("Max", (np.log(max_p), c))
        plt.bar(np.log(min_p), c, color=color, width=0.5)
        plt.annotate("Min", (np.log(min_p), c))


if __name__ == "__main__":

    N = int(sys.argv[1])
    smoothing_type = sys.argv[2]

    MyTokeniser = Tokeniser()
    F = Functions()

    sentences = MyTokeniser.clean_data()
    sentences = [MyTokeniser.add_buffer(N, sentence)
                 for sentence in MyTokeniser.clean_data()]
    sentences, word_freq, total_words = MyTokeniser.dense_clean(sentences)

    # for sentence in sentences[:10]:
    #     l = " ".join(sentence)
    #     print(l)

    # file1 = open(f"./results/{sys.argv[3][5:]}-clean.txt", "a")
    # for sentence in sentences:
    #     l = "".join(sentence)
    #     file1.write(f"{l}\n")
    # file1.close()

    test_set = sentences[:1000]
    train_set = sentences[1000:]
    # train_set = input()

    grams = [F.create_ngram_dictionary(i, train_set) for i in range(1, 5)]
    grams.reverse()

    print([len(grams[i]) for i in range(len(grams))])
    print()

    fig = plt.figure(figsize=(10, 5))

    if smoothing_type == 'k':
        F.plot_graph(F.run('Train', train_set, N,
                     F.kneser_ney), 'blue', 'Train')
        F.plot_graph(F.run('Test', test_set, N, F.kneser_ney),
                     'maroon', 'Test')
    elif smoothing_type == 'w':
        F.plot_graph(F.run('Train', train_set, N,
                     F.witten_bell), 'blue', 'Train')
        F.plot_graph(F.run('Test', test_set, N, F.witten_bell),
                     'maroon', 'Test')
    else:
        F.plot_graph(F.run('Train', train_set, N,
                     F.simple_smoothing), 'blue', 'Train')
        F.plot_graph(F.run('Test', test_set, N, F.simple_smoothing),
                     'maroon', 'Test')

    plt.xlabel("log(Perplexity)")
    plt.ylabel("Valid Sentences")
    plt.title("Performance Comparision")
    plt.legend()

    # plt.show()
