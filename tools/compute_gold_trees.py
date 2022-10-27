import pickle
from tqdm import tqdm
import os

def extract_spans_and_labels(tree, idx=0):
    spans = [[idx, idx+len(tree.leaves())-1]]
    labels = [tree.label()]
    start_idx = idx
    for node in tree:
        if len(node.leaves()) > 1:
            node_span, node_label = extract_spans_and_labels(node,start_idx)
            spans.extend(node_span)
            labels.extend(node_label)
        start_idx += len(node.leaves())
    return spans, labels

def preprocess_pentathlon_datasets():
    import benepar
    benepar.download('benepar_en3')
    parser = benepar.Parser("benepar_en3")
    for root_dir in [
        "data/DiDeMo/challenge-release-2/",
        "data/YouCook2/challenge-release-2/",
        "data/MSRVTT/challenge-release-2/"
    ]:
        captions = pickle.load(open(os.path.join(root_dir,'processed-captions.pkl'), 'rb'))
        if 'challenge-release-1' in root_dir:
            val_vids = [l.strip() for l in open(os.path.join(root_dir, 'public_server_val.txt')).readlines()]
            captions = {vid: captions[vid] for vid in val_vids}

        sent2tree = {}
        for vid, sentences in tqdm(captions.items()):

            gold_trees = list(parser.parse_sents([benepar.InputSentence(words=words) for words in sentences]))
            spans, labels = [], []
            for tree in gold_trees:
                assert tree.label() == 'TOP'
                tree = tree[0]
                span, label = extract_spans_and_labels(tree)
                spans.append(span)
                labels.append(label)
            for sent, tree, span, label in zip(sentences, gold_trees, spans, labels):
                sent2tree.update({' '.join(sent): {'tree': str(tree), 'span': span, 'label': label}})

        pickle.dump(sent2tree, open(os.path.join(root_dir,"non_binary_tree.pkl"), 'wb'))

if __name__ == '__main__':
    preprocess_pentathlon_datasets()