import os

class CharTokenizer:

    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.chars = []

    def build_vocab(self, text):
        new_chars = sorted(list(set(text)))

        if self.chars:
            existing_set = set(self.chars)

            for ch in new_chars:
                if ch not in existing_set:
                    self.chars.append(ch)
        else:
            self.chars = new_chars

        self._rebuild_maps()
        return self.chars

    def load_vocab(self, path):
        if not os.path.exists(path):
            return

        chars = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ch = line.rstrip("\n")
                if ch == "\\n":
                    ch = "\n"
                chars.append(ch)

        self.chars = chars
        self._rebuild_maps()

    def save_vocab(self, path):
        directory = os.path.dirname(os.path.abspath(path))

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for ch in self.chars:
                if ch == "\n":
                    f.write("\\n\n")
                else:
                    f.write(ch + "\n")

    def _rebuild_maps(self):
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return "".join([self.itos[i] for i in tokens])
    
    def getvocabsize(self):
        return len(self.chars)