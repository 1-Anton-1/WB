import numpy as np
import pandas as pd
import torch
from data_loader import get_embeddings

# Перемещаем модель на GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_closest_chunk(question, df=pd.read_pickle("data/df_knowledge_base_prepared.pkl")):
    # Получаем эмбеддинг для вопроса
    question_embedding = get_embeddings(question).reshape(1, -1)

    # Извлекаем эмбеддинги из DataFrame и перемещаем их на GPU
    embeddings = np.vstack(df['embeddings'].values)
    embeddings = torch.tensor(embeddings).to(device)

    # Перемещаем эмбеддинг вопроса на GPU
    question_embedding = torch.tensor(question_embedding).to(device)

    # Вычисляем косинусное сходство между вопросом и каждым chunk
    similarities = torch.nn.functional.cosine_similarity(question_embedding, embeddings)

    # Находим индекс chunk с максимальным сходством
    closest_idx = torch.argmax(similarities).item()

    return df.iloc[closest_idx]['chunk']


# print(find_closest_chunk('Какая минимальная площадь ПВЗ?'))