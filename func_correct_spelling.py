import requests  # Импортируем модуль requests для отправки HTTP-запросов.

def correct_spelling(text: str) -> str:
    """
    Функция для исправления опечаток с помощью YandexSpeller.
    """
    api_url = 'https://speller.yandex.net/services/spellservice.json/checkText'  # Задаем URL для запроса к YandexSpeller.
    params = {'text': text}  # Создаем словарь параметров для запроса, включая текст, который нужно проверить на опечатки.
    response = requests.get(api_url, params=params)  # Отправляем GET-запрос к YandexSpeller с текстом для проверки.
    flag = False
    if response.status_code == 200:  # Проверяем успешность запроса (код ответа 200 означает успешный запрос).
        corrected_text = text  # Инициализируем переменную corrected_text начальным текстом.
        for correction in response.json():  # Проходим по каждой исправленной ошибке, предоставленной YandexSpeller.
            if correction['s']:  # Проверяем, есть ли исправление для данного слова.
                corrected_text = corrected_text.replace(correction['word'], correction['s'][0])  # Заменяем слово с опечаткой на исправленное.
        flag = True
        return corrected_text, flag  # Возвращаем текст с исправленными опечатками.
    else:  # Если запрос не был успешным (код ответа не 200), то
        return text, flag  # возвращаем исходный текст без изменений.