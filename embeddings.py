import warnings
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