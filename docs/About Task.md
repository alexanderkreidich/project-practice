# Структра [/src](https://github.com/alexanderkreidich/project-practice/tree/master/src)
```markdawn
/ src
├── dataset/
|   └── spam.csv # Данные для обучения модели
├── models/
|   ├──  SmapHamModel.joblib #обученная модель классификации сообщений на спам/не спам
|   └──  TfVModel.joblib  #обученная модель для преобразования сообщения в форму в числа
├── main.py  # Исполнительный файл
├── ChooseClassifier.py # Файл с выбором лучших моделей для обучения
├── TrainModel.py # Файл с обучением выбранных моделей
└── requirements.txt    # Список зависимостей
```

# Описание
- SMS-классификатор – это модель машинного обучения, предназначенная для автоматического определения категории текстовых сообщений (спам / не спам).
- Как работает
  - Пользователь отправляет SMS-сообщение через веб-интерфейс.
  - Модель на основе алгоритма SVC анализирует текст и возвращает результат с вероятностью принадлежности к спаму.
- Преимущества
  - Высокая точность классификации (98.2%).
  - Простота интеграции благодаря экспорту модели в joblib.
  - Удобный Flask-интерфейс для тестирования.

# Функционал
## Основные модули
- ChooseClassifier.py
    - Сравнение алгоритмов (SVC, RandomForest, Naive Bayes).
    - Выбор оптимальной модели (SVC + TfidfVectorizer).
    - /first_aid [травма] - Получить инструкции по оказанию первой помощи при травме (например, при ожоге, порезе).
- TrainModel.py
  - Векторизация текста (TfidfVectorizer).
  - Обучение и сохранение модели (joblib).
- main.py
  - Flask-приложение для проверки сообщений в реальном времени.

# Возможности системы
- Классификация SMS по категориям (спам / не спам).
- Возврат вероятности принадлежности к спаму.

# Изучение технологии
- Ресурсы:
  - [Scikit-learn документация](https://scikit-learn.org/stable/user_guide.html)
  - [документация Flask](https://flask.palletsprojects.com/en/stable/)
  - [документация библиотеки joblib](https://joblib.readthedocs.io/en/stable/)
## Этап 1: Выбор оптимальной модели для обучения
- Реализация
  - Функция для перебора списка моделей
  - Выбор лучшей модели из того, что вывела функция
```python
def perform(classifiers, vectorizers, atributes, target) -> None:
    for classifier in classifiers:
      for vectorizer in vectorizers:
        string = ''
        string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

        # train
        X_train, X_test, y_train, y_test = train_test_split(atributes, target,
                                                            test_size=0.2, random_state=42)

        vectorize_text = vectorizer.fit_transform(X_train)

        classifier.fit(vectorize_text, y_train)

        # score
        vectorize_text = vectorizer.transform(X_test)
        score = classifier.score(vectorize_text, y_test)
        string += '. Has score: ' + str(score)
        print(string)
```
## Этап 2: Обучение выбранной модели
Основные компоненты:
1) Загружаем наш датасет и инициализируем выбранные модели:
```python
#Загружаем датасет и разделяем его на тренеровочные и тестовые части
data = pd.read_csv('dataset/spam.csv', encoding='latin-1')
X, y = data["v2"],data["v1"]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2, random_state=42)
#Инициализируем SVC и TfidfVectorizer
clf = SVC(kernel='linear', probability=True)
TfV = TfidfVectorizer()
```
2) Обучаем модели SVC и TfidfVectorizer на выделенных тренеровочных данных, затем тестируем спам-детектор на тестовых данных:
```python
#Трансформируем текст в числа и обучаем модель
vectorize_text = TfV.fit_transform(X_train)
clf.fit(vectorize_text, y_train)

#Смотрим метрики для получившейся модели
vectorize_text_test = TfV.transform(X_test)
report = classification_report(y_test, clf.predict(vectorize_text_test),
                               target_names=['ham', 'spam'])
print(report)

dump(clf, 'Models/SmapHamModel.joblib')
dump(TfV, 'Models/TfVModel.joblib')
```
В конце закидываем модели в .joblib файл
## Этап 3: Разработка Flask-интерфейса
```python
@app.route('/', methods=['GET'])
def index():
    global clf
    global TfV
    message = request.args.get('message', '')
    error = ''
    predict_proba = ''
    predict = ''

    try:
        if len(message) > 0:
            vectorize_message = TfV.transform([message])
            predict = clf.predict(vectorize_message)[0]
            predict_proba = clf.predict_proba(vectorize_message).tolist()

    except BaseException as inst:
        error = str(type(inst).__name__) + ' ' + str(inst)

    return jsonify(message=message, predict_proba=predict_proba, predict=predict, error=error)
```
