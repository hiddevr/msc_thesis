from fastDamerauLevenshtein import damerauLevenshtein
import random
from tqdm import tqdm
from collections import defaultdict


class SequenceMatcher:

    def __init__(self):
        self.index = defaultdict(set)
        self.index_values = list()
        self.nearest_indices = dict()

    def _fit_train_sequences(self, train_sequences):
        sorted_sequences = sorted(train_sequences, key=len)
        for sequence in sorted_sequences:
            self.index[len(sequence)].add(sequence)
        self.index_values = sorted(list(self.index.keys()))

        for seq_key in self.index.keys():
            seq_lst = list(self.index[seq_key])
            random.shuffle(seq_lst)
            self.index[seq_key] = set(seq_lst)

    def _fit_test_indices(self, test_sequences):
        for query_sequence in test_sequences:
            query_length = len(query_sequence)
            nearest_index = min(self.index_values, key=lambda x: abs(x - query_length))
            self.nearest_indices[query_sequence] = nearest_index

    def fit(self, train_sequences, test_sequences):
        self._fit_train_sequences(train_sequences)
        self._fit_test_indices(test_sequences)
        return

    def find_match(self, query_sequence):
        nearest_index = self.nearest_indices[query_sequence]
        max_distance = 100000
        matched_sequence = None
        non_distance_update_count = 0

        for potential_match in self.index[nearest_index]:
            distance = damerauLevenshtein(query_sequence, potential_match, similarity=False)
            if distance < max_distance:
                max_distance = distance
                matched_sequence = potential_match
            else:
                non_distance_update_count += 1
                if non_distance_update_count >= 100:
                    return matched_sequence, max_distance
        return matched_sequence, max_distance