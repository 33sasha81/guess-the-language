import os
import gdown

# Список файлов и их ссылок
file_links = {
    "eng_w.txt": "https://drive.google.com/uc?id=1MOdBYmKczmadRDLH1sMUYbBWtWhTFp9O",
    "rus_w.txt": "https://drive.google.com/uc?id=1H257C5Nf3-IsQ41HeNpYaPhgGXXGnrjU",
    "deu_w.txt": "https://drive.google.com/uc?id=1BKcdCKrYwy00XklRdf5FVkHYCOtOfgJF",
    "spa_w.txt": "https://drive.google.com/uc?id=19Rcyk1zZOuai9Oh8N5f0szvjAW8lQzeY",
    "fra_w.txt": "https://drive.google.com/uc?id=12kMibFwZCxSfiyQczdbX4bYay3tCJvx8"
}

# Создаём папку для файлов, если её нет
os.makedirs("data", exist_ok=True)

# Загружаем файлы
for filename, url in file_links.items():
    output_path = os.path.join("data", filename)
    if not os.path.exists(output_path):  # Проверяем, скачан ли файл
        gdown.download(url, output_path, quiet=False)
print()
print("Все файлы загружены!")

