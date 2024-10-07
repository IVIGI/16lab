from flask import Flask
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
from model.neuron import SingleNeuron


app = Flask(__name__)

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