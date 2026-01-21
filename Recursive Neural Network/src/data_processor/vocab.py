from collections import Counter
import torch

class Vocab:
    "Build vocab, encode, decode text to indices"
    def __init__(self, min_freq=1):
        self.word2idx = {"<PAD>": 0,
                         "<UNK>": 1}
        self.idx2word = {0: "<PAD>",
                         1: "<UNK>"}
        self.label2idx = {}
        self.idx2label = {}
        self.min_freq = min_freq

    def build_vocab(self, list_of_sentences):
        "Build vocab from a list of tokenized sentences"
        all_tokens = [token for sentence in list_of_sentences for token in sentence]
        counts = Counter(all_tokens)

        for word, freq in counts.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"Vocab size: {len(self.word2idx)}")
        self.vocab_size = len(self.word2idx)

    def build_label_vocab(self, labels):
        "Build label vocab (for both sentiment & ner tags)"
        unique_labels = set()
        for item in labels:
            if isinstance(item, list): # for NER tags
                for label in item:
                    unique_labels.add(label)
            else:
                unique_labels.add(item)

        for label in sorted(list(unique_labels)):
            if label not in self.label2idx:
                idx = len(self.label2idx)
                self.label2idx[label] = idx
                self.idx2label[idx] = label

        print(f"Label vocab size: {len(self.label2idx)}")
        self.num_labels = len(self.label2idx)
        print(f"Labels: {self.label2idx}")

    def encode_sentence(self, tokens, max_len=100):
        "Covert tokens into indices"
        indices = [self.word2idx.get(token
        , self.word2idx["<UNK>"]) for token in tokens]
        if len(indices) < max_len:
            indices += [self.word2idx["<PAD>"]] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return torch.tensor(indices, dtype=torch.long)

    def encode_label(self, label_input, max_len=100):
        """
        Convert label(s) to indices.
        - If label_input is a string: Text Classification (Task 1, 2)
        - If label_input is a list: NER (Task 3)
        """
        if isinstance(label_input, str):
            return torch.tensor(self.label2idx[label_input], dtype=torch.long)
        elif isinstance(label_input, list):
            label_ids = [self.label2idx[l] for l in label_input]
            if len(label_ids) < max_len:
                label_ids += [-100] * (max_len - len(label_ids))
            else:
                label_ids = label_ids[:max_len]
        return torch.tensor(label_ids, dtype=torch.long)

    def decode_label(self, label_ids):
        if isinstance(label_ids, torch.Tensor):
            label_ids = label_ids.tolist()
        if isinstance(label_ids, int): # for text classification
            return self.idx2label[label_ids]
        return [self.idx2label[idx] for idx in label_ids if idx != -100] # for ner tags

if __name__ == "__main__":
    vocab_vsfc = Vocab()
    # VSFC dataset
    print("Test for VSFC Dataset")
    vsfc_tokens = [["giáo_trình", "hay"], ["giảng_viên", "nhiệt_tình"]]
    vsfc_labels = ["positive", "positive"]
    vocab_vsfc.build_vocab(vsfc_tokens)
    vocab_vsfc.build_label_vocab(vsfc_labels)

    print(f"Encoded vsfc sentence:", {vocab_vsfc.encode_sentence(["giáo_trình", "tuyệt_vời"], max_len=5)})
    print(f"Encode label:", {vocab_vsfc.encode_label("positive")})

    # Phoner dataset
    vocab_phoner = Vocab()
    print("Test for PhoNER Dataset")
    phoner_tokens = [["Bộ", "Y_tế", "dập", "dịch"], ["COVID", "nguy_hiểm"]]
    phoner_labels = [["B-ORG", "I-ORG", "O", "O"], ["B-MISC", "O"]]
    vocab_phoner.build_vocab(phoner_tokens)
    vocab_phoner.build_label_vocab(phoner_labels)
    print(f"Encoded ner tags:", {vocab_phoner.encode_label(["B-ORG", "I-ORG", "O", "O"])})