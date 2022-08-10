from scipy.spatial import distance
from gensim.models import Word2Vec
import torch
from data import CleanData
from cbow import CBOW
import time

start = time.time()

data = CleanData()
data.initialise()

# model = CBOW(734) # 10 samples
# model = CBOW(22815)  # 10000 samples
# model = CBOW(38075)  # 30000 samples
model = CBOW(37463)  # 30000 samples cleaned
model.load_state_dict(torch.load(f'./results/model-19'))
# model.load_state_dict(torch.load(f'./results/new_model'))
model.eval()


check_word = 'camera'
camera_embedding = model.get_word_emdedding(
    check_word, data.word2index).detach()
print(camera_embedding)

cosine_disances = []
for i, word in enumerate(data.vocab):
    word_embedding = model.get_word_emdedding(word, data.word2index).detach()
    cosine_disances.append(
        (1-distance.cosine(camera_embedding, word_embedding), i))

cosine_disances.sort(key=lambda x: x[0], reverse=True)

print(f'\n{check_word}')
for score, i in cosine_disances[1:11]:
    print(f'{data.index2word[i]} : {score}')
print()


new_model = Word2Vec(data.corpus, min_count=1, vector_size=10, window=5, sg=0)
print(new_model)
print(new_model.wv[check_word])
for tpl in new_model.wv.most_similar(check_word)[:10]:
    print(tpl)
    # print(tpl, tpl[0] in data.vocab)
print()

end = time.time()
print(f"Time Elapsed: {end - start}\n")
