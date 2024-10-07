import tensorflow as tf

import numpy as np

X_class = np.array([[2, 0.6, 5],
              [4, 1.2, 0.7],
              [1.8, 0.6, 0.5],
              [5, 1.3, 0.7],
              [2.2, 0.8, 0.5],
              [3, 1, 0.5],
              [5, 1.1, 0.8]])
y_class = np.array([1, 0, 1, 0, 1, 1, 0])

# Создание модели для классификации
model_class = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(3,)),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Один выход для бинарной классификации
])

model_class.compile(optimizer='adam', loss='binary_crossentropy')

# Обучение модели
model_class.fit(X_class, y_class, epochs=200, batch_size=32)

# Прогноз
test_data = np.array([[9, 9, 0.7]])
y_pred_class = model_class.predict(test_data)
print("Предсказанные значения:", y_pred_class, *np.where(y_pred_class >= 0.5, 'Седан', 'Внедорожник'))
# Сохранение модели для классификации
model_class.save('classification_model.h5')