import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Загрузка и подготовка данных
def load_data(file_path):
    """Загрузка CSV-файла с вопросами и ответами"""
    try:
        df = pd.read_csv(file_path)

        required_columns = {'question', 'answer'}
        if not required_columns.issubset(df.columns):
            raise ValueError("CSV должен содержать колонки 'question' и 'answer'")

        if 'category' not in df.columns:
            df['category'] = 'general'

        # Очистка данных
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        print(f"Загружено {len(df)} вопросов-ответов")
        print("Пример данных:")
        print(df.head())

        return df

    except Exception as e:
        print(f"Ошибка загрузки файла: {e}")
        return None


#Обучение модели
def train_model(df):
    """Обучение модели классификации вопросов"""
    # Векторизация текста
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=['какие', 'как', 'есть', 'ли', 'для', 'на', 'с', 'что', 'это'],
        ngram_range=(1, 2))

    X = vectorizer.fit_transform(df['question'])  # Исправленные кавычки
    y = df['category']

    #Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    #Инициализация моделей
    models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('nb', MultinomialNB()),
        ('svm', SVC(kernel='linear', probability=True))
    ]

    #Ансамблевая модель
    ensemble = VotingClassifier(estimators=models, voting='soft')
    ensemble.fit(X_train, y_train)

    #Оценка модели
    y_pred = ensemble.predict(X_test)
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred))

    #Визуализация
    plt.figure(figsize=(10, 6))
    sns.countplot(y=y, order=y.value_counts().index)
    plt.title("Распределение категорий вопросов")
    plt.show()

    return ensemble, vectorizer


#Сохранение модели
def save_model(model, vectorizer, model_dir='models'):
    """Сохранение модели и векторизатора"""
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, 'admission_bot_model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    print(f"\nМодель сохранена в папке '{model_dir}'")


def main():
    csv_file = "data/faq_datasets/admission_faq.csv.txt"

    print("Загрузка данных...")
    faq_data = load_data(csv_file)
    if faq_data is None:
        return

    #Обучение модели
    print("\nОбучение модели...")
    model, vectorizer = train_model(faq_data)

    # 3. Сохранение модели
    save_model(model, vectorizer)

    print("\nОбучение завершено успешно!")


if __name__ == "__main__":
    main()