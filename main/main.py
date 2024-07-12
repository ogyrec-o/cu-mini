import json
import cuMiniModel as model

# Загрузка данных и подготовка для обучения
intents = model.load_data('intents.json')
words, classes, documents = model.preprocess_data(intents)
train_x, train_y = model.create_training_data(words, classes, documents)

# Создание и обучение модели
input_shape = len(train_x[0])
output_shape = len(train_y[0])
cuMiniModel = model.create_model(input_shape, output_shape)
cuMiniModel = model.train_model(cuMiniModel, train_x, train_y)

# Сохранение модели
model.save_model(cuMiniModel, 'model/cuMiniModel.keras')

# Пример использования модели для чат-бота
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    ints = model.predict_class(message, cuMiniModel, words, classes)
    res = model.get_response(ints, intents)
    print(f"Bot: {res}")
