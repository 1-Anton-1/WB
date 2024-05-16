from get_answer import get_answer_openchat

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time

app = FastAPI()

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Инициализируем шаблоны
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/query")
async def query(question: str):
    start_time = time.time()  # Записываем начальное время
    answer = get_answer_openchat(question)  # Выполняем запрос
    end_time = time.time()  # Записываем конечное время
    execution_time = end_time - start_time  # Вычисляем время выполнения
    execution_time = round(execution_time, 2)  # Округляем до двух знаков после запятой
    return {
        "answer": answer,
        "execution_time": execution_time
    }