import pickle
import gzip
import os
from collections import defaultdict
from typing import List, Dict


class BigramLanguageModel:
    def __init__(self, words: List[str]):
        """Инициализация объекта модели."""
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.vocab_size = 0

        # Подсчёт униграмм и биграмм из входного списка слов
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            self.bigram_counts[w1][w2] += 1
            self.unigram_counts[w1] += 1
            self.unigram_counts[w2] += 1

        self.vocab_size = len(self.unigram_counts)

    def probability(self, word: str, prev_word: str) -> float:
        """Метод вычисляет вероятность появления слова word после prev_word."""
        if prev_word in self.bigram_counts and word in self.bigram_counts[prev_word]:
            return self.bigram_counts[prev_word][word] / self.unigram_counts[prev_word]
        elif word in self.unigram_counts:
            return self.unigram_counts[word] / self.vocab_size
        else:
            return 1 / self.vocab_size

    def to_dict(self):
        unigram_counts = dict(self.unigram_counts)
        bigram_counts = {k: dict(v) for k, v in self.bigram_counts.items()}
        return {
            'unigram_counts': unigram_counts,
            'bigram_counts': bigram_counts,
            'vocab_size': self.vocab_size
        }

    @classmethod
    def from_dict(cls, data):
        """Создаёт объект BigramLanguageModel из словаря, полученного при десериализации."""
        model = cls([])
        model.unigram_counts = defaultdict(int, data['unigram_counts'])
        model.bigram_counts = defaultdict(lambda: defaultdict(int))
        for k, v in data['bigram_counts'].items():
            model.bigram_counts[k] = defaultdict(int, v)
        model.vocab_size = data['vocab_size']
        return model


def load_corpus(path: str) -> List[str]:
    """Загружает текстовый корпус из файла и возвращает список слов."""
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as file:
            text = file.read()
    else:
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()

    sentences = text.splitlines()
    words = []
    for sentence in sentences:
        words.extend(sentence.split())
    return words


def train_and_save_models(corpus_paths: Dict[str, str], model_save_paths: Dict[str, str]):
    """Обучает биграммные модели для каждого языка и сохраняет их на диск."""
    for lang, path in corpus_paths.items():
        if not os.path.exists(path):
            print(f"Ошибка: Файл корпуса для языка {lang} не найден по пути {path}")
            return

    word_lists = {lang: load_corpus(path) for lang, path in corpus_paths.items()}

    models = {lang: BigramLanguageModel(words) for lang, words in word_lists.items()}

    for lang, model in models.items():
        with open(model_save_paths[lang], 'wb') as f:
            pickle.dump(model.to_dict(), f)
        print(f"Модель для языка {lang} сохранена в {model_save_paths[lang]}")


corpus_paths = {
    'English': 'eng_w.txt',
    'Russian': 'rus_w.txt',
    'German': 'deu_w.txt',
    'Spanish': 'spa_w.txt',
    'French': 'fra_w.txt'
}

model_save_paths = {
    'English': 'eng_model.pkl',
    'Russian': 'rus_model.pkl',
    'German': 'deu_model.pkl',
    'Spanish': 'spa_model.pkl',
    'French': 'fra_model.pkl'
}

if __name__ == "__main__":
    train_and_save_models(corpus_paths, model_save_paths)
