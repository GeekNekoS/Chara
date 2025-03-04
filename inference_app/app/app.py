from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import  os


app = Flask(__name__)

# Загрузка модели
MODEL_PATH = "../../models/2/model.keras"
# MODEL_PATH = "models/2/model.keras"
# Получаем абсолютный путь относительно текущего скрипта
# MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", '..', MODEL_PATH)
print(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# Убедитесь, что `RandomFlip` не влияет на инференс, удалите его из preprocessing


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получаем входные данные
        input_data = request.json.get("input")
        if input_data is None:
            return jsonify({"error": "No input data provided"}), 400

        # Преобразуем данные в numpy-формат
        input_array = np.array(input_data, dtype=np.float32)

        # Предсказание
        predictions = model.predict(input_array).tolist()
        return jsonify({"predictions": predictions}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
