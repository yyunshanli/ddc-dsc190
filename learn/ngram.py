import random
import pickle
import json
import numpy as np
from collections import Counter
import argparse
from math import log, exp


class NgramSequence:
    def __init__(self, chart_notes):
        self.sequence = [sym for _, _, _, sym in chart_notes]

    def get_ngrams(self, k, pre=True, post=True):
        prepend = ['<pre{}>'.format(i) for i in reversed(range(k - 1))] if pre else []
        append = ['<post>'] if post else []
        sequence = prepend + self.sequence + append
        return (tuple(sequence[i:i + k]) for i in range(len(sequence) - k + 1))


class NgramLanguageModel:
    def __init__(self, k, ngram_counts):
        self.k = k
        self.ngram_counts = ngram_counts
        self.history_counts = Counter({ngram[:-1]: count for ngram, count in ngram_counts.items()})
        self.vocab = set(w for ngram in ngram_counts for w in ngram)
        self.vocab_size = len(self.vocab)

    def mle(self, ngram):
        return self.ngram_counts[ngram] / self.history_counts[ngram[:-1]]

    def laplace(self, ngram, smooth=1):
        numerator = self.ngram_counts.get(ngram, 0) + smooth
        denominator = self.history_counts.get(ngram[:-1], 0) + self.vocab_size + smooth
        return numerator / denominator

    def generate(self, history, strategy='argmax'):
        if strategy == 'argmax':
            max_prob = max(self.laplace(history + (v,)) for v in self.vocab)
            candidates = [v for v in self.vocab if self.laplace(history + (v,)) == max_prob]
            return random.choice(candidates)
        raise NotImplementedError()


def train_ngram_model(dataset_files, k, diff_filter, model_fp):
    ngram_counts = Counter()
    for file in dataset_files:
        with open(file, 'r') as f:
            song_meta = json.load(f)
        for chart in song_meta['charts']:
            if diff_filter and diff_filter != chart['difficulty_coarse']:
                continue
            for ngram in NgramSequence(chart['notes']).get_ngrams(k):
                ngram_counts[ngram] += 1

    model = NgramLanguageModel(k, ngram_counts)
    with open(model_fp, 'wb') as f:
        pickle.dump(model, f)


def evaluate_ngram_model(dataset_files, model_fp, k, diff_filter):
    with open(model_fp, 'rb') as f:
        model = pickle.load(f)

    entropies, accuracies = [], []
    for file in dataset_files:
        with open(file, 'r') as f:
            song_meta = json.load(f)
        for chart in song_meta['charts']:
            if diff_filter and diff_filter != chart['difficulty_coarse']:
                continue

            log_prob, hits, n = 0.0, 0, 0
            for ngram in NgramSequence(chart['notes']).get_ngrams(k, pre=True, post=False):
                generated = model.generate(ngram[:-1])
                if generated == ngram[-1]:
                    hits += 1
                log_prob += log(model.laplace(ngram))
                n += 1

            entropies.append(-log_prob / n)
            accuracies.append(hits / n)

    perplexities = np.exp(entropies)
    results = { 
        'cross_entropy': (np.mean(entropies), np.std(entropies)),
        'perplexity': (np.mean(perplexities), np.std(perplexities)),
        'accuracy': (np.mean(accuracies), np.std(accuracies))
    }

    for metric, (mean, std) in results.items():
        print(f"{metric.capitalize()}: Mean={mean:.4f}, Std={std:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_fp', type=str, help='Path to dataset file')
    parser.add_argument('model_fp', type=str, help='Path to save/load the model')
    parser.add_argument('--k', type=int, default=1, help='N-gram size')
    parser.add_argument('--diff', type=str, help='Difficulty filter')
    parser.add_argument('--task', type=str, choices=['train', 'eval'], default='train', help='Task to perform')

    args = parser.parse_args()
    with open(args.dataset_fp, 'r') as f:
        dataset_files = f.read().split()

    if args.task == 'train':
        train_ngram_model(dataset_files, args.k, args.diff, args.model_fp)
    elif args.task == 'eval':
        evaluate_ngram_model(dataset_files, args.model_fp, args.k, args.diff)