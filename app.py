from flask import Flask, render_template, request
import pickle
import time
import os

app = Flask(__name__)

# -------------------------------
# Load Model
# -------------------------------
try:
    model = pickle.load(open('model/model.pkl', 'rb'))
    vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

    if os.path.exists('model/model_name.pkl'):
        model_name = pickle.load(open('model/model_name.pkl', 'rb'))
    else:
        model_name = "Best Model"

    print("✅ Model Loaded:", model_name)

except Exception as e:
    print("❌ Error loading model:", e)

# -------------------------------
# Home Route
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# -------------------------------
# Prediction Route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('news_text', '')

        if not text.strip():
            return render_template('index.html', error="Please enter some text!")

        start_time = time.time()

        vectorized = vectorizer.transform([text])
        prediction = model.predict(vectorized)[0]

        # Result
        result = "REAL NEWS ✅" if prediction == "REAL" else "FAKE NEWS ❌"

        # Metrics
        process_speed = round(time.time() - start_time, 3)
        word_count = len(text.split())
        read_time = max(1, word_count // 200)

        return render_template(
            'index.html',
            prediction=result,
            original_text=text,
            word_count=word_count,
            read_time=read_time,
            process_speed=process_speed,
            model_name=model_name
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)