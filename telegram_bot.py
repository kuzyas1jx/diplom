import telebot
import joblib

# Загрузка модели и векторизатора
model = joblib.load('models/admission_bot_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

API_TOKEN = '8117630693:AAEv3IBbv3bmClNetb6YNwqZmHh2G1WvxqU'
bot = telebot.TeleBot(API_TOKEN)


# Команда /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(
        message.chat.id,
        "Привет! Я интеллектуальный чат-бот для абитуриентов. "
        "Задайте свой вопрос или введите команду /help для справки."
    )


# Команда /help
@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = (
        "📌 *Доступные команды:*\n"
        "/start – запустить бота\n"
        "/help – список команд\n\n"
        "Вы также можете просто задать свой вопрос, и я определю его категорию."
    )
    bot.send_message(message.chat.id, help_text, parse_mode="Markdown")


# Обработка произвольных текстовых сообщений
@bot.message_handler(func=lambda message: True)
def classify_and_respond(message):
    question = message.text
    vector = vectorizer.transform([question])
    category = model.predict(vector)[0]

    response = f"📂 Определённая категория: *{category}*\n\n(Ответ будет предоставлен по этой категории.)"
    bot.send_message(message.chat.id, response, parse_mode="Markdown")

# Запуск бота
print("Бот запущен...")
bot.infinity_polling()