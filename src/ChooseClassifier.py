from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.model_selection import train_test_split
import pandas


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

# open data-set and divide it
data = pandas.read_csv('dataset/spam.csv', encoding='latin-1')
X, y = data["v2"],data["v1"]

perform(
    [
        BernoulliNB(),
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        AdaBoostClassifier(),
        BaggingClassifier(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier(),
        CalibratedClassifierCV(),
        DummyClassifier(),
        PassiveAggressiveClassifier(),
        RidgeClassifier(),
        RidgeClassifierCV(),
        SGDClassifier(),
        SVC(kernel='linear'),
        LogisticRegression(),
        KNeighborsClassifier()
    ],
    [
        CountVectorizer(),
        TfidfVectorizer(),
        HashingVectorizer()
    ],
    X,
    y
)