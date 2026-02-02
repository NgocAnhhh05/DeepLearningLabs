from collections import Counter
import torch

class Vocab:
    """
    Manages vocabulary for NMT. Handles word-to-index and index-to-word mappings.
    Includes special tokens: <PAD>, <UNK>, <SOS>, <EOS>.
    """
    def __init__(self, min_freq=1):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.min_freq = min_freq

    def build_vocab(self, list_of_sentences):
        """
        Builds vocab from a list of tokenized sentences.
        list_of_sentences: List[List[str]]
        """
        tokens = [token for sent in list_of_sentences for token in sent]
        counts = Counter(tokens)

        for word, freq in counts.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"Vocab built. Total size: {len(self.word2idx)}")

    def encode(self, tokens, max_len=None, add_sos=False, add_eos=False):
        """
        Converts tokens to indices. Adds SOS/EOS if required and pads/truncates.
        """
        indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        if add_sos: indices = [self.word2idx["<SOS>"]] + indices
        if add_eos: indices = indices + [self.word2idx["<EOS>"]]

        if max_len:
            if len(indices) < max_len:
                indices += [self.word2idx["<PAD>"]] * (max_len - len(indices))
            else:
                indices = indices[:max_len]
        return indices

    def decode(self, indices):
        """Converts indices back to tokens, skipping special padding/SOS tokens."""
        special_ids = [self.word2idx["<PAD>"], self.word2idx["<SOS>"], self.word2idx["<EOS>"]]
        return [self.idx2word[idx] for idx in indices if idx not in special_ids]

if __name__ == "__main__":
    # Test Vocab
    v = Vocab()
    v.build_vocab([["i", "love", "ai"], ["machine", "learning"]])
    ids = v.encode(["i", "love", "ai"], max_len=10, add_sos=True, add_eos=True)
    print(f"Encoded: {ids}")
    print(f"Decoded: {v.decode(ids)}")