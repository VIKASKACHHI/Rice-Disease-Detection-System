from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = Flask(__name__)

# Load the model (ensure you use the correct path and format)
model = tf.keras.models.load_model('E:/Rice crop Health/rice_disease_model1.h5')  # Replace with .h5 model

# Local disease-to-treatment mapping and reasons
disease_info = {
    'Blast Disease': {
        'treatment_hindi': 'ट्राइसीक्लाज़ोल जैसे कवकनाशी का प्रयोग करें। उचित जल निकासी सुनिश्चित करें।',
        'treatment_english': 'Apply fungicide like Tricyclazole. Ensure proper water drainage.',
        'reason_hindi': 'यह रोग आमतौर पर नमी और उच्च आर्द्रता वाले क्षेत्रों में फैलता है।'
    },
    'Brown Spot': {
        'treatment_hindi': 'मैंकोजेब जैसे कवकनाशी का उपयोग करें। नाइट्रोजन प्रबंधन में सुधार करें।',
        'treatment_english': 'Use fungicides like Mancozeb. Improve nitrogen management.',
        'reason_hindi': 'यह रोग पोषण की कमी और गीले पत्तों की वजह से होता है।'
    },
    'Bacterial Blight': {
        'treatment_hindi': 'कॉपर आधारित बैक्टीरिसाइड्स का छिड़काव करें। प्रतिरोधी चावल की किस्मों का उपयोग करें।',
        'treatment_english': 'Spray copper-based bactericides. Use resistant rice varieties.',
        'reason_hindi': 'यह रोग उच्च नमी और संक्रमित बीजों की वजह से फैलता है।'
    },
    'False Smut': {
        'treatment_hindi': 'कार्बेंडाजिम जैसे कवकनाशी का उपयोग करें। फसल की स्वच्छता बनाए रखें।',
        'treatment_english': 'Use carbendazim fungicide. Maintain crop cleanliness.',
        'reason_hindi': 'यह रोग अत्यधिक नमी और अस्वच्छ खेतों की वजह से होता है।'
    }
}

# Helper function to get disease information
def get_disease_info(disease):
    info = disease_info.get(disease, {})
    return {
        'treatment_hindi': info.get('treatment_hindi', 'उपचार की जानकारी उपलब्ध नहीं है।'),
        'treatment_english': info.get('treatment_english', 'No treatment information available.'),
        'reason_hindi': info.get('reason_hindi', 'रोग का कारण अज्ञात है।')
    }

# Image preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    input_size = model.input_shape[1:3]  # Get the height and width from the model's input shape
    image = image.resize(input_size)  # Resize to the model's expected input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image_bytes = file.read()
    image = preprocess_image(image_bytes)

    # Predict disease
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    disease_classes = ['Blast Disease', 'Brown Spot', 'Bacterial Blight', 'False Smut']
    predicted_disease = disease_classes[predicted_class]

    # Get disease information
    disease_info = get_disease_info(predicted_disease)

    return jsonify({
        'disease': predicted_disease,
        'treatment_hindi': disease_info['treatment_hindi'],
        'treatment_english': disease_info['treatment_english'],
        'reason_hindi': disease_info['reason_hindi']
    })

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
