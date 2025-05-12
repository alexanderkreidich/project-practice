from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from joblib import dump
import pandas as pd

#Загружаем датасет и разделяем его на тренеровочные и тестовые части
data = pd.read_csv('dataset/spam.csv', encoding='latin-1')
X, y = data["v2"],data["v1"]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2, random_state=42)
#Инициализируем класификатор и TfidfVectorizer
clf = SVC(kernel='linear', probability=True)
TfV = TfidfVectorizer()

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


