FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем PyTorch с поддержкой CUDA напрямую
RUN pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

COPY . .
# Устанавливаем переменную окружения для запуска Uvicorn
ENV PYTHONPATH=/app
# Открываем порт для доступа к приложению
EXPOSE 8000
# Команда для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]