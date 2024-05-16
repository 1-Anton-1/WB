import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


# Загружаем модель эмбеддера и токенайзер
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model_name = 'intfloat/multilingual-e5-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

# Перемещаем модель на GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

# Считываем файлы с google-диска
df_knowledge_base = pd.read_excel(f"https://drive.google.com/uc?id=1fCvwaS-u4MtSo8z_vqzsfJGc8D9_ZEn0")
df_QA_pairs = pd.read_excel(f"https://drive.google.com/uc?id=136c244E3ihfmAu3znJvEPR9WngEXl5Om")
df_BQA = pd.read_excel(f"https://drive.google.com/uc?id=1MHRAMqzZUCmNgScfUs9LZJmmLJszDkFL")


df_knowledge_base['embeddings'] = df_knowledge_base['chunk'].apply(get_embeddings)  # Получение эмбеддингов для каждого chunk

# Приведение всех эмбеддингов к одной длине
min_length = min(len(embed) for embed in df_knowledge_base['embeddings'])
df_knowledge_base['embeddings'] = df_knowledge_base['embeddings'].apply(lambda x: x[:min_length])


# Функция для сохранения DataFrame в формат .pkl
def save_df_to_pickle(df, filename):
    df.to_pickle(f"data/{filename}.pkl")

# Сохранение DataFrame
save_df_to_pickle(df_knowledge_base, "df_knowledge_base_prepared")
save_df_to_pickle(df_QA_pairs, "df_QA_pairs")
save_df_to_pickle(df_BQA, "df_BQA")