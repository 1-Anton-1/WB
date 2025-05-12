import warnings
import pandas as pd
import os
import gdown
from embeddings import get_embeddings  # Изменено

def prepare_and_save_data():
    # Создание директории 'WB/data', если она не существует
    if not os.path.exists('WB/data'):
        os.makedirs('WB/data')

    # Скачиваем файлы
    url_BQA = 'https://drive.google.com/uc?id=1iMOskRbKgbBIrSPb8kKqzKHufcRDGZcO'
    url_knowledge_base = 'https://drive.google.com/uc?id=1fCvwaS-u4MtSo8z_vqzsfJGc8D9_ZEn0'
    url_QA_pairs = 'https://drive.google.com/uc?id=136c244E3ihfmAu3znJvEPR9WngEXl5Om'

    print("Скачивание файлов...")
    gdown.download(url_BQA, 'WB/data/BQA.xlsx', quiet=False)
    gdown.download(url_knowledge_base, 'WB/data/knowledge_base.xlsx', quiet=False)
    gdown.download(url_QA_pairs, 'WB/data/QA_pairs.xlsx', quiet=False)

    # Читаем файлы
    print("Чтение файлов...")
    df_BQA = pd.read_excel('WB/data/BQA.xlsx')
    df_knowledge_base = pd.read_excel('WB/data/knowledge_base.xlsx')
    df_QA_pairs = pd.read_excel('WB/data/QA_pairs.xlsx')

    print("Создание эмбеддингов...")
    df_knowledge_base['embeddings'] = df_knowledge_base['chunk'].apply(get_embeddings)

    # Приведение всех эмбеддингов к одной длине
    min_length = min(len(embed) for embed in df_knowledge_base['embeddings'])
    df_knowledge_base['embeddings'] = df_knowledge_base['embeddings'].apply(lambda x: x[:min_length])

    # Сохранение DataFrame
    print("Сохранение данных...")
    df_knowledge_base.to_pickle("WB/data/df_knowledge_base_prepared.pkl")
    df_QA_pairs.to_pickle("WB/data/df_QA_pairs.pkl")
    df_BQA.to_pickle("WB/data/df_BQA.pkl")

    # Удаляем временные Excel файлы
    os.remove('WB/data/BQA.xlsx')
    os.remove('WB/data/knowledge_base.xlsx')
    os.remove('WB/data/QA_pairs.xlsx')
    print("Готово!")

if __name__ == "__main__":
    prepare_and_save_data()