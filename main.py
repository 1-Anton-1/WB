from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from func_correct_spelling import correct_spelling
from get_answer import get_answer_openchat

import pandas as pd
import time
import data_loader

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Загружаем данные из .pkl файлов
df_knowledge_base = pd.read_pickle("data/df_knowledge_base_prepared.pkl")
df_QA_pairs = pd.read_pickle("data/df_QA_pairs.pkl")
df_BQA = pd.read_pickle("data/df_BQA.pkl")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Получаем случайные вопросы
    random_questions_QA_pairs = df_QA_pairs['question'].sample(3).tolist()  # 3 случайных вопроса из df_QA_pairs
    random_questions_BQA = df_BQA['question'].sample(3).tolist()  # 3 случайных вопросов из df_BQA
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
