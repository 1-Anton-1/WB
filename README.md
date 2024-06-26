# 5-ый этап

## Release ver.1.0:

### Описание текущих файлов:

    - /static - директория с файлами стилистики:

        - style.css - стили элементов сайта, указанных в файле index.html

    - /templates - директория для html файлов:

        - index.html - файл с html разметкой сайта

    - main.py - основной исполняемый файл

    - data_loader.py - файл для загрузки исходных данных (базы знаний, базы вопросов/ответов)

    - model.py - файл загрузки модели с HF

    - retriever.py - поиск нужных чанок из локальной базы знаний

    - requirements.txt - файл с перечнем библиотек и их версий

    - func_correct_spelling.py - файл с функцией устранения опечаток

### Инструкция по сборке:

#### Клонирование репозитория

    git clone https://github.com/1-Anton-1/WB.git
    cd WB

#### Сборка Docker-образа

    docker build -t qa_system_app .

#### Запуск Docker-контейнера

    После сборки образа запустите контейнер:

    docker run -p 8000:8000 qa_system_app .

#### Доступ к приложению

    После запуска контейнера приложение будет доступно по адресу:

    http://localhost:8000

    

P. S. Если не запускается Docker (localhost:8000) (для запуска требуется 5-10 мин, т. к. скачиватеся LLM и помещается на GPU, (если возможно)), то безотказно работает команда в терминале pycharm: uvicorn main:app --reload 

P. S. S. В сборке есть также два файла для запуска с Jupyter Notebook auto_com_t5_v4.ipynb и результаты работы модели из ноутбука df_score_llm_openchat_2.xlsx



## Update ver.1.1:

	1. Добавлено создание пустой директории 'data' для последующего сохранения dataframe в формате .pkl.

## Update ver.2.0:

	1. Добавлен файл func_correct_spelling.py - для устранения опечаток.

	2. Из файла index.html убран код определяющий стиль интерфейса web app. Код был перенесен в файл static/style.css.

	3. В файле static/style.css добавлено подсветка действий пользователя (изменение рамки и фона поля ввода, изменение цвета кнопки "Отправить").

	4. Скорректирован параметр температуры LLM 0.7->0.6 для уменьшения галлюцинаций (например, "Я не знаю ответа на ваш вопрос ...", хотя LLM сформулировала ответ на основе базы знаний).

	5. Добавлен вывод 3-х случайных вопросов из исходной базы вопросов и ответов экспертов (df_QA_pairs), для копирования этих вопросов в поле ввода и визуальной оценки качества сгенерированного моделью ответа на основе найденной информации из базы знаний.

	6. Добавлен вывод 3-х случайных вопросов из базы вопросов-ответов (df_BQA), для демонстрации работы полной QA-system с модулем обработки уже известных вопросов (оптимизация затрат вычислительных мощностей).