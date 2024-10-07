import tensorflow as tf

import numpy as np

X_train    = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
y_train = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)


# Создание модели для регрессии
model_reg = tf.keras.Sequential([
    tf.keras.layers.Dense(4,  input_shape=(1,)),
    tf.keras.layers.Dense(3, ),
    tf.keras.layers.Dense(1, activation='linear')  # Один выход для регрессии
])

model_reg.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model_reg.fit(X_train, y_train, epochs=500)

# Прогноз
print(model_reg.predict(np.array([100])))
# Сохранение модели для регрессии
model_reg.save('regression_model.h5')