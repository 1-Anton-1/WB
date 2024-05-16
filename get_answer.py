from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from retriever import find_closest_chunk
from model import model, tokenizer, pipe_openchat


import torch
import warnings

# Подавляем все предупреждения
warnings.filterwarnings("ignore")


def get_answer_openchat(question, max_new_tokens=512, temperature=0.8):
    """Функция для генерации ответом с помощью LLM openchat-3.5"""
    s = 'Фрагменты базы знаний:\n'
    s += find_closest_chunk(question)
    content = """Ты – сотрудник техподдержки Вайлдберриз. Начинай свой ответ с слова «Здравствуйте!» \
    Ниже приведены фрагменты базы знаний и запрос пользователя. Ответь на вопрос пользователя используя информацию из фрагментов ниже.\
    Если в фрагментах ниже не будет ответа скажи: «Я не знаю ответ, пожалуйста сформулируйте свой вопрос иначе».""" + s
    return pipe_openchat([{'role': 'system',
                           'content': content},
                          {'role': 'user', 'content': question}],
                         max_new_tokens=max_new_tokens,
                         do_sample=True,
                         temperature=temperature)[0]['generated_text'][-1]['content'][1:]