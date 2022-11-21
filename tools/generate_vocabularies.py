import pickle
import os
from collections import Counter
import json
from tqdm import tqdm
from glob import glob
def obtain_all_captions():
    all_captions = {}
    for p in glob(f"data/HowTo100M/processed_captions/*.pickle"):
        all_captions.update(pickle.load(open(p, 'rb')))
    return all_captions

def generate_howto100m_vocabularies():
    data_root = "data/HowTo100M"
    counter_path = os.path.join(data_root, 'counter.pkl')
    captions = obtain_all_captions()
    if not os.path.exists(counter_path):
        counter = Counter()
        for vid, item in tqdm(captions.items()):
            for sent in item:
                for word in sent['description']:
                    counter[word] += 1
        pickle.dump(counter, open(os.path.join(data_root, 'counter.pkl'), 'wb'))

    counter = pickle.load(open(os.path.join(data_root, 'counter.pkl'), 'rb'))
    sorted_words = [i[0] for i in sorted(counter.items(), key=lambda x: x[1], reverse=True)][:20000]
    word2int = {w: i for i, w in enumerate(sorted_words)}
    int2word = {i: w for i, w in enumerate(sorted_words)}
    pickle.dump(int2word, open(os.path.join(data_root, 'processed-int2word-20k.pkl'), 'wb'))
    pickle.dump(word2int, open(os.path.join(data_root, 'processed-word2int-20k.pkl'), 'wb'))


if __name__ == '__main__':
    generate_howto100m_vocabularies()
