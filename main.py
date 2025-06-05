import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Загрузка и подготовка данных
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8-sig')

        required_columns = {'Вопрос', 'Ответ'}
        if not required_columns.issubset(df.columns):
            raise ValueError("CSV должен содержать колонки 'Вопрос' и 'Ответ'")

        if 'Категория' not in df.columns:
            df['Категория'] = 'general'

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        print(f"Загружено {len(df)} записей")
        print("Пример данных:")
        print(df.head())

        return df

    except Exception as e:
        print(f"Ошибка загрузки файла: {e}")
        return None


# Обучение модели классификации
def train_model(df):
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=['какие', 'как', 'есть', 'ли', 'для', 'на', 'с', 'что', 'это'],
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df['Вопрос'])
    y = df['Категория']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('nb', MultinomialNB()),
        ('svc', SVC(kernel='linear', probability=True))
    ], voting='soft')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(10, 6))
    sns.countplot(y=y, order=y.value_counts().index)
    plt.title("Распределение категорий вопросов")
    plt.show()

    return model, vectorizer


# Сохранение модели и векторизатора
def save_model(model, vectorizer, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'admission_bot_model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    print(f"\nМодель сохранена в папке '{model_dir}'")


# Классификация нового вопроса
def classify_question(model, vectorizer, question):
    vec = vectorizer.transform([question])
    prediction = model.predict(vec)[0]
    return prediction


# Основная логика
def main():
    csv_file = "data/faq_datasets/admision_FAQ.csv"

    print("Загрузка данных...")
    df = load_data(csv_file)
    if df is None:
        return

    print("\nОбучение модели...")
    model, vectorizer = train_model(df)

    save_model(model, vectorizer)

    example = "Когда начинается прием документов?"
    print("\nПример классификации:")
    print("Вопрос:", example)
    print("Категория:", classify_question(model, vectorizer, example))

    print("\nОбучение завершено успешно!")


if __name__ == "__main__":
    main()