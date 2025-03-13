import pickle
import tkinter as tk
from math import exp, log
from typing import List, Dict
from collections import defaultdict
import os


class BigramLanguageModel:
    def __init__(self, words: List[str]):
        """Инициализация модели."""
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.vocab_size = 0

    def probability(self, word: str, prev_word: str) -> float:
        """Вычисляет вероятность появления слова word после prev_word."""
        if prev_word in self.bigram_counts and word in self.bigram_counts[prev_word]:
            return self.bigram_counts[prev_word][word] / self.unigram_counts[prev_word]
        elif word in self.unigram_counts:
            return self.unigram_counts[word] / self.vocab_size
        else:
            return 1 / self.vocab_size

    @classmethod
    def from_dict(cls, data):
        """Создаёт объект модели из словаря, загруженного из файла."""
        model = cls([])
        model.unigram_counts = defaultdict(int, data['unigram_counts'])
        model.bigram_counts = defaultdict(lambda: defaultdict(int))
        for k, v in data['bigram_counts'].items():
            model.bigram_counts[k] = defaultdict(int, v)
        model.vocab_size = data['vocab_size']
        return model


def calculate_perplexity(model: BigramLanguageModel, test_words: List[str]) -> float:
    """Вычисляет перплексию текста для оценки качества модели."""
    if len(test_words) < 2:
        raise ValueError("Для вычисления перплексии нужно хотя бы два слова")
    total_log_prob = 0.0
    for i in range(1, len(test_words)):
        prev_word = test_words[i - 1]
        word = test_words[i]
        prob = model.probability(word, prev_word)
        if prob > 0:
            total_log_prob += log(prob)
        else:
            total_log_prob += log(1e-10)
    perplexity = exp(-total_log_prob / (len(test_words) - 1))
    return perplexity


def load_models(model_paths: Dict[str, str]) -> Dict[str, BigramLanguageModel]:
    """Загружает биграммные модели для каждого языка из файлов."""
    models = {}
    for lang, path in model_paths.items():
        if not os.path.exists(path):
            print(f"Ошибка: Файл модели для языка {lang} не найден по пути {path}")
            continue
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                models[lang] = BigramLanguageModel.from_dict(model_data)
            print(f"Модель для языка {lang} загружена из {path}")
        except Exception as e:
            print(f"Ошибка при загрузке модели для языка {lang}: {e}")
    return models


def identify_language(test_words: List[str], models: Dict[str, BigramLanguageModel]) -> tuple:
    """Определяет язык текста, сравнивая перплексии для всех моделей."""
    if not models:
        return "Не удалось загрузить модели.", {}
    try:
        perplexities = {}
        for lang, model in models.items():
            ppx = calculate_perplexity(model, test_words)
            perplexities[lang] = ppx
        identified_lang = min(perplexities, key=perplexities.get)
        return identified_lang, perplexities
    except ValueError as e:
        return f"Ошибка: {e}", {}


model_paths = {
    'English': 'eng_model.pkl',
    'Russian': 'rus_model.pkl',
    'German': 'deu_model.pkl',
    'Spanish': 'spa_model.pkl',
    'French': 'fra_model.pkl'
}


models = load_models(model_paths)


def create_gui():
    root = tk.Tk()
    root.title("Определение языка текста")
    root.geometry("800x600")
    root.configure(bg="pink")

    input_frame = tk.Frame(root, bg="pink")
    input_frame.pack(pady=10, fill=tk.BOTH, expand=True)

    tk.Label(input_frame, text="Введите предложение:", font=("Arial", 14), bg="pink").pack(anchor="w")

    text_frame = tk.Frame(input_frame, bg="white", bd=2, relief="solid")
    text_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=15)

    text_entry = tk.Text(text_frame, width=50, height=5, font=("Arial", 14), wrap=tk.WORD, bd=0, bg="white")
    scrollbar = tk.Scrollbar(text_frame, command=text_entry.yview)
    text_entry.configure(yscrollcommand=scrollbar.set)

    text_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    result_frame = tk.Frame(root, bg="white", bd=2, relief="solid")
    result_frame.pack(pady=20, fill=tk.BOTH, expand=True, padx=40)

    result_text_widget = tk.Text(result_frame, height=6, width=50, font=("Arial", 14), bd=0, bg="white", state=tk.DISABLED)
    result_text_widget.pack(fill=tk.BOTH, expand=True)

    def on_identify():
        test_text = text_entry.get("1.0", tk.END).strip()
        test_words = test_text.split()

        if len(test_words) < 2:
            result_text_widget.configure(state=tk.NORMAL)
            result_text_widget.delete(1.0, tk.END)
            result_text_widget.insert(tk.END, "Ошибка: Введите хотя бы два слова.")
            result_text_widget.configure(state=tk.DISABLED)
        else:
            identified_lang, perplexities = identify_language(test_words, models)
            if identified_lang:
                result_text_widget.configure(state=tk.NORMAL)
                result_text_widget.delete(1.0, tk.END)
                result_text_widget.insert(tk.END, f"Определённый язык: {identified_lang}\n\n", "large")
                result_text_widget.insert(tk.END, "Перплексии:\n", "normal")
                sorted_perplexities = sorted(perplexities.items(), key=lambda x: x[1])
                for lang, ppx in sorted_perplexities:
                    result_text_widget.insert(tk.END, f"{lang}: {ppx:.4f}\n", "small")
                result_text_widget.configure(state=tk.DISABLED)

    def clear_entry():
        text_entry.delete("1.0", tk.END)
        result_text_widget.configure(state=tk.NORMAL)
        result_text_widget.delete("1.0", tk.END)
        result_text_widget.configure(state=tk.DISABLED)

    button_frame = tk.Frame(root, bg="pink")
    button_frame.pack(pady=10)

    identify_button = tk.Button(button_frame, text="Определить язык", command=on_identify, font=("Arial", 12), bg="pink", fg="black")
    identify_button.pack(side=tk.LEFT, padx=5)

    clear_button = tk.Button(button_frame, text="Очистить", command=clear_entry, font=("Arial", 12), bg="pink", fg="black")
    clear_button.pack(side=tk.LEFT, padx=5)

    root.mainloop()


if __name__ == "__main__":
    if models:
        create_gui()
    else:
        print("Не удалось загрузить модели. GUI не будет запущен.")
