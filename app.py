from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the models
model1 = load_model('model1.h5')
model2 = load_model('model2.h5')
models = {'Model 1': model1, 'Model 2': model2}

# Function to load and preprocess image
def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Adjust target size to match your model's input
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Handle model selection
            selected_model_name = request.form.get('model_selector')
            selected_model = models[selected_model_name]

            # Prepare image and predict
            image = prepare_image(filepath)
            predictions = selected_model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            
            # Remove uploaded file after processing
            os.remove(filepath)
            
            return render_template('result.html', predicted_class=predicted_class, model_name=selected_model_name)

    return render_template('index.html', models=models.keys())

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
