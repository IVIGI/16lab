import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify, render_template
from model.neuron import SingleNeuron

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"},
        {"name": "Лаба 4", "url": "p_lab4"},
        {"name": "Лаба 18", "url": "p_lab18"},]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Загрузка весов из файла
new_neuron = SingleNeuron(input_size=2)
new_neuron.load_weights('model/neuron_weights.txt')
model_reg = tf.keras.models.load_model('model/regression_model.h5')
model_class = tf.keras.models.load_model('model/classification_model.h5')
model_cloth = tf.keras.models.load_model('model/model_cloth.h5')

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные ФИО", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_knn.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + pred)

@app.route("/p_lab2")
def f_lab2():
    return render_template('lab2.html', title="Логистическая регрессия", menu=menu)


@app.route("/p_lab3")
def f_lab3():
    return render_template('lab3.html', title="Логистическая регрессия", menu=menu)

@app.route("/p_lab4", methods=['POST', 'GET'])
def p_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Первый нейрон", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2'])]])
        predictions = new_neuron.forward(X_new)
        print("Предсказанные значения:", predictions, *np.where(predictions >= 0.5, 'Помидор', 'Огурец'))
        return render_template('lab4.html', title="Первый нейрон", menu=menu,
                               class_model="Это: " + str(*np.where(predictions >= 0.5, 'Помидор', 'Огурец')))


@app.route("/p_lab18", methods=['POST', 'GET'])
def p_lab18():
    if request.method == 'GET':
        return render_template('lab18.html', title="Определение одежды", menu='', class_model='')

    if request.method == 'POST':
        # Get the uploaded image
        img_file = request.files['image']
        img_path = img_file.filename
        img_file.save(img_path)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Make predictions
        predictions = model_cloth.predict(img)
        predicted_class = class_names[np.argmax(predictions)]

        return render_template('lab18.html', title="Определение одежды", menu='', class_model="Это: " + predicted_class)

@app.route('/api', methods=['get'])
def get_sort():
    X_new = np.array([[float(request.args.get('sepal_length')),
                       float(request.args.get('sepal_width')),
                       float(request.args.get('petal_length')),
                       float(request.args.get('petal_width'))]])
    pred = loaded_model_knn.predict(X_new)

    return jsonify(sort=pred[0])

@app.route('/api_v2', methods=['get'])
def get_sort_v2():
    request_data = request.get_json()
    X_new = np.array([[float(request_data['sepal_length']),
                       float(request_data['sepal_width']),
                       float(request_data['petal_length']),
                       float(request_data['petal_width'])]])
    pred = loaded_model_knn.predict(X_new)

    return jsonify(sort=pred[0])


@app.route('/api_reg', methods=['get'])
def predict_regression():
    # Получение данных из запроса http://localhost:5000/api_reg?celsius=100
    input_data = np.array([float(request.args.get('celsius'))])
    # input_data = np.array(input_data.reshape(-1, 1))

    # Предсказание
    predictions = model_reg.predict(input_data)

    return jsonify(fahrenheit=str(predictions))

@app.route('/api_class', methods=['get'])
def predict_classification():
    # Получение данных из запроса http://localhost:5000/api_class?width=5&length=5
    input_data = np.array([[float(request.args.get('power')),
                            float(request.args.get('fuel')),
                            float(request.args.get('seats'))]])
    print(input_data)
    # input_data = np.array(input_data.reshape(-1, 1))

    # Предсказание
    predictions = model_class.predict(input_data)
    print(predictions)
    result = 'Седан' if predictions >= 0.5 else 'Внедорожник'
    print(result)
    # меняем кодировку
    app.config['JSON_AS_ASCII'] = False
    return jsonify(kuzov = str(result))

if __name__ == "__main__":
    app.run(debug=True)
