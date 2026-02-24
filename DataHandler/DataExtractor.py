from pypdf import PdfReader
import os
import re
import random
import unicodedata

class DataExtractor:

    def get_from_pdf(self, path):
        if not os.path.exists(path):
            return ""

        reader = PdfReader(path)
        pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                cleaned = self.clean_text(text)
                pages.append(cleaned)

        return "\n\n".join(pages)

    def clean_text(self, text):
        text = unicodedata.normalize("NFKC", text)

        text = "".join(
            ch for ch in text
            if unicodedata.category(ch)[0] != "C"
        )

        text = "".join(
            ch for ch in text
            if ch == "\n" or 32 <= ord(ch) <= 126
        )

        text = re.sub(r'\n\d+\n', '\n', text)

        text = re.sub(r'[ \t]+', ' ', text)

        text = re.sub(r'SECTION\s+[IVXLC]+', '', text)

        return text.strip()

class DatasetBuilder:

    def __init__(self, chunk_size=1000, seed=42):
        self.chunk_size = chunk_size
        self.samples = []
        random.seed(seed)

    def add_text(self, text, domain_tag):
        text = text.strip()
        sentences = re.split(r'(?<=[.!?]) +', text)

        buffer = ""

        for sentence in sentences:
            if len(buffer) + len(sentence) < self.chunk_size:
                buffer += " " + sentence
            else:
                self._add_chunk(buffer.strip(), domain_tag)
                buffer = sentence

        if len(buffer) > 200:
            self._add_chunk(buffer.strip(), domain_tag)

    def _add_chunk(self, chunk, domain_tag):
        formatted = f"<{domain_tag}>\n{chunk}\n<END>\n"
        self.samples.append(formatted)

    def shuffle(self):
        random.shuffle(self.samples)

    def split(self, train_ratio=0.9):
        split_index = int(len(self.samples) * train_ratio)
        return self.samples[:split_index], self.samples[split_index:]

    def save(self, samples, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(samples))