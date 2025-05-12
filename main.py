from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import warnings
import torch
from transformers import AutoTokenizer, AutoModel
import os

from func_correct_spelling import correct_spelling
from get_answer import get_answer_openchat

import pandas as pd
import time

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static"
)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

print("Загрузка модели...")
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

print("Загрузка данных...")
# Загружаем данные из .pkl файлов
df_knowledge_base = pd.read_pickle("WB/data/df_knowledge_base_prepared.pkl")
print("База знаний загружена")
df_QA_pairs = pd.read_pickle("WB/data/df_QA_pairs.pkl")
print("QA пары загружены")
df_BQA = pd.read_pickle("WB/data/df_BQA.pkl")
print("BQA загружены")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Получаем случайные вопросы
    random_questions_QA_pairs = df_QA_pairs['question'].sample(3).tolist()
    random_questions_BQA = df_BQA['question'].sample(3).tolist()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "random_questions_QA_pairs": random_questions_QA_pairs,
        "random_questions_BQA": random_questions_BQA
    })

@app.get("/query")
async def get_query(question: str):
    start_time = time.time()
    question, question_flag = correct_spelling(question)
    # Ищем вопрос в df_BQA
    if question in df_BQA['question'].values:
        answer_row = df_BQA[df_BQA['question'] == question].iloc[0]
        answer = answer_row['answer']
        message = "Ответ найден в базе вопросов-ответов"
        expert_answer = None
    elif question in df_QA_pairs['question'].values:
        answer_row = df_QA_pairs[df_QA_pairs['question'] == question].iloc[0]
        answer, answer_flag = correct_spelling(get_answer_openchat(question))
        message = ""
        expert_answer = answer_row['answer']
    else:
        answer, answer_flag = correct_spelling(get_answer_openchat(question))
        message = ""
        expert_answer = None
    execution_time = time.time() - start_time
    return {"message": message, "answer": answer, "execution_time": execution_time, "expert_answer": expert_answer}

if __name__ == "__main__":
    import uvicorn
    print("Запуск сервера...")
    uvicorn.run(app, host="0.0.0.0", port=8000)