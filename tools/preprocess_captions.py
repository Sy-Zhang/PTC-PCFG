import pickle
import os
from tqdm import tqdm
import json
from glob import glob
from collections import defaultdict, Counter
import string

import spacy
nlp = spacy.load('en_core_web_md')

def sentence_to_token(sentence, discard_punct=True):
    tokens = []
    for token in nlp.tokenizer(sentence.lower()):
        if token.is_punct and discard_punct:
            continue
        if token.is_digit:
            tokens.append('N')
        else:
            tokens.append(token.text)
    return tokens

def preprocess_challenge_captions():
    for root_dir in [
        'data/DiDeMo/challenge-release-2/',
        'data/YouCook2/challenge-release-2/',
        'data/MSRVTT/challenge-release-2/',
    ]:
        raw_captions = pickle.load(open(os.path.join(root_dir, 'raw-captions.pkl'), 'rb'))
        processed_captions = defaultdict(list)
        for vid, captions in raw_captions.items():
            for caption in captions:
                if 'wallcan' in caption:
                    idx = caption.index('wallcan')
                    caption[idx] = 'wall'
                    caption.insert(idx+1, 'can')
                caption = ' '.join(caption)
                tokens = sentence_to_token(caption)
                processed_captions[vid].append(tokens)
        pickle.dump(processed_captions, open(os.path.join(root_dir, 'processed-captions.pkl'), 'wb'))

def preprocess_howto100m_captions(pickle_id):
    import re
    import theano
    import theano.tensor as T
    from collections import deque
    from punctuator2 import models
    from punctuator2.play_with_model import punctuate

    x = T.imatrix('x')

    print("Loading model parameters...")
    net, _ = models.load("tools/punctuator2/Demo-Europarl-EN.pcl", 1, x)

    print("Building model...")
    predict = theano.function(inputs=[x], outputs=net.y)
    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary
    reverse_word_vocabulary = {v:k for k,v in net.x_vocabulary.items()}
    reverse_punctuation_vocabulary = {v:k for k,v in net.y_vocabulary.items()}

    print('Loading data ...')
    captions = pickle.load(open(f"data/HowTo100M/raw_captions/0{pickle_id}.pickle", 'rb'))

    def find_common_prefix_postfix(s1, s2):
        size = 0
        for i in reversed(range(max(0, len(s1) - len(s2)), len(s1) - 1)):
            if s1[i:] == s2[:len(s1) - i]:
                size = len(s1) - i
        return size
    def process(k, v):

        all_word_starts, all_word_ends, all_words = deque(), deque(), deque()

        v['text'] = [re.sub(r"\n|-|;|,|:|\"|#|$|%|&|\(|\)|\*|\+|,|-|/|:|;|<|=|>|@|[|\\|]|^|_|`|{|\||}|~", " ", s) for s in v['text'] if isinstance(s, str)]
        v['text'] = [re.sub("\!\!+", " ! ", re.sub("\?\?+", " ? ", re.sub("\.\.+", " . ", s))) for s in v['text']]
        v['text'] = [re.sub(" +", " ", s) for s in v['text']]
        v['text'] = [s.lstrip().rstrip() for s in v['text']]
        i = 0
        processed_text = []
        processed_start = []
        processed_end = []
        while i < len(v['text']):
            text, start, end = v['text'][i], v['start'][i], v['end'][i]
            j = i + 1
            while j < len(v['text']):
                size = find_common_prefix_postfix(text, v['text'][j])
                if size > 0:
                    text = text + v['text'][j][size:]
                    end = v['end'][j]
                    j += 1
                else:
                    break

            processed_text.append(text)
            processed_start.append(start)
            processed_end.append(end)
            i = j
        v['text'], v['start'], v['end'] = processed_text, processed_start, processed_end

        for start, end, subtitle in zip(v['start'], v['end'], v['text']):
            words = sentence_to_token(subtitle, discard_punct=False)
            if len(words) > 0:
                duration = end - start
                time_unit = duration / (len(words))
                word_starts = [start + time_unit * i for i in range(len(words))]
                word_ends = [start + time_unit * (i + 1) for i in range(len(words))]
                all_word_starts.extend(word_starts)
                all_word_ends.extend(word_ends)
                all_words.extend(words)

        other_punct_pattern = r'\.+|\-+'
        paragraph = " ".join([w for w in all_words if w in ['.', '?', '!'] or not (w in string.punctuation or re.match(other_punct_pattern, w))])
        is_punct = ' .' in paragraph or ' ?' in paragraph or ' !' in paragraph
        if is_punct:
            sentences = [s.lstrip().rstrip().split(" ") for s in re.split(r' \.\.\. | \.\.\.| \. | \.| \!| \!| \? | \?', paragraph) if s.lstrip().rstrip() != '']
            sentences = [[t for t in tokens if t not in string.punctuation] for tokens in sentences if len(tokens) > 0]
        else:
            out = punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary,
                            reverse_word_vocabulary, paragraph, True)
            sentences = re.split(r'\.PERIOD|\?QUESTIONMARK|\!EXCLAMATIONMARK', out)
            sentences = [
                sent.replace(',COMMA', '').replace('-DASH', '').replace(':COLON', '').replace(";SEMICOLON", '').replace('<UNK>', 'unk')
                .lstrip().rstrip() for sent in sentences
            ]
            sentences = [sentence_to_token(re.sub(" +", " ", s), discard_punct=False) for s in sentences if s.lstrip().rstrip() != '']
            sentences = [tokens for tokens in sentences if len(tokens) > 0]
        sentence_starts, sentence_ends = [], []
        processed_sentences = []
        for tokens in sentences:
            start, end = all_word_starts[-1], all_word_ends[0]
            sent = []
            for word in tokens:
                popped_word = all_words.popleft()
                popped_start = all_word_starts.popleft()
                popped_end = all_word_ends.popleft()
                while popped_word in string.punctuation or re.match(other_punct_pattern, popped_word):
                    popped_word = all_words.popleft()
                    popped_start = all_word_starts.popleft()
                    popped_end = all_word_ends.popleft()
                if word != 'unk':
                    try:
                        assert word == popped_word
                    except:
                        print(word, popped_word)
                start = min(start, popped_start)
                end = max(end, popped_end)
                sent.append(popped_word)
            processed_sentences.append(sent)
            sentence_starts.append(start)
            sentence_ends.append(end)

        output = {k: {'start': sentence_starts, 'end': sentence_ends, 'text': processed_sentences}}
        for sent in processed_sentences:
            print(" ".join(sent))
        json.dump(output, open(f"data/HowTo100M/processed_caption/{k}.json", 'w'))

    for i, (k, v) in enumerate(tqdm(captions.items())):
        if not os.path.exists(f"data/HowTo100M/processed_caption/{k}.json"):
            try:
                process(k, v)
            except:
                print(k)
                break

    new_annotations = {}
    for k in tqdm(captions.keys()):
        processed_caption = json.load(open(f"data/HowTo100M/processed_caption/{k}.json"))
        start, end, text = processed_caption[k]['start'], processed_caption[k]['end'], processed_caption[k]['text']
        new_annotations[k] = [{'time': [s, e], 'description': t} for s, e, t in zip(start, end, text)]
    os.makedirs("data/HowTo100M/processed_captions", exist_ok=True)
    pickle.dump(new_annotations, open(f'data/HowTo100M/processed_captions/0{pickle_id}.pickle', 'wb'))
    return new_annotations

def separate_raw_json():
    import json
    from tqdm import tqdm
    raw_captions = json.load(open("data/HowTo100M/raw_caption.json"))
    raw_captions = [(k, v) for k, v in raw_captions.items()]

    for i in tqdm(range(40)):
        captions = {k: v for k, v in raw_captions[i::40]}
        json.dump(captions, open(f"data/HowTo100M/raw_caption/raw_caption_{i}.json", 'w'))

if __name__ == '__main__':
    # Preprocess captions in DiDeMo, YouCook2 and MSRVTT
    preprocess_challenge_captions()

    # Preprocess HowTo100M subtitles
    for pickle_id in range(8):
        preprocess_howto100m_captions(pickle_id)

