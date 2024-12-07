from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained rice disease detection model
model = load_model('rice_disease_model2.h5')
class_labels = ['Bacterial Blight', 'Blast', 'Brown Spot', 'False Smut', 'Healthy', 'Tungro']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def predict_disease(file_path):
    # Load the image and resize it to match the model's expected input size
    img = image.load_img(file_path, target_size=(128, 128))  # Change target_size to (128, 128)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

    # Make a prediction
    prediction = model.predict(img_array)

    # Get the class label with the highest probability
    predicted_class_index = np.argmax(prediction[0])
    return class_labels[predicted_class_index]

@app.route('/')
def index():
    return render_template('index.html')  # Serve the index.html file

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    file.save(file_path)

    # Predict the disease
    disease = predict_disease(file_path)

    # Define treatment and cause for each disease
    disease_info = {
        'Bacterial Blight': {
            'treatment_hindi': "बैक्टीरियल ब्लाइट से बचने के लिए चावल के पौधों को साफ रखें और सही कीटनाशकों का उपयोग करें।",
            'treatment_english': "To prevent bacterial blight, keep rice plants clean and use appropriate pesticides.",
            'cause': "यह बैक्टीरिया ज़ैंथोमोनस ओरायजे के कारण होता है।",
            'medicine_image': "/static/copper_bactericide.png"
        },
        'Blast': {
            'treatment_hindi': "फंगल संक्रमण को रोकने के लिए उचित फफूंदनाशकों का छिड़काव करें।",
            'treatment_english': "Spray appropriate fungicides to prevent fungal infection.",
            'cause': "यह कवक मैग्नापोर्थे ओरायजे के कारण होता है।",
            'medicine_image': "/static/tricyclazole.png"
        },
        'Brown Spot': {
            'treatment_hindi': "पोटाशियम और फॉस्फेट खाद का उपयोग करें और संक्रमित पौधों को हटा दें।",
            'treatment_english': "Use potassium and phosphate fertilizers and remove infected plants.",
            'cause': "यह कवक कोक्लियोबोलस मियाबेनस के कारण होता है।",
            'medicine_image': "/static/mancozeb.png"
        },
        'False Smut': {
            'treatment_hindi': "पौधों को संक्रमण से बचाने के लिए उचित फफूंदनाशकों का उपयोग करें।",
            'treatment_english': "Use appropriate fungicides to protect plants from infection.",
            'cause': "यह कवक उस्टिलागिनोइडिया विरेंस के कारण होता है।",
            'medicine_image': "/static/flase_smut.png"
        },
        'Healthy': {
            'treatment_hindi': "चावल के पौधे स्वस्थ हैं, किसी उपचार की आवश्यकता नहीं है।",
            'treatment_english': "The rice plant is healthy; no treatment is needed.",
            'cause': "कोई रोग नहीं पाया गया।",
            'medicine_image': "/static/healthy.png"
        },
        'Tungro': {
            'treatment_hindi': "संक्रमित पौधों को हटा दें और कीटनाशकों का उपयोग करें।",
            'treatment_english': "Remove infected plants and use insecticides.",
            'cause': "यह तुंगरो वायरस के कारण होता है, जो पत्ती के टिड्डों द्वारा फैलता है।",
            'medicine_image': "/static/tungro_medicine.png"
        }
    }

    # Get details for the predicted disease
    details = disease_info.get(disease, {})

    # Respond with prediction and treatment info
    return jsonify({
        'disease': disease,
        'treatment_hindi': details.get('treatment_hindi', 'No information available'),
        'treatment_english': details.get('treatment_english', 'No information available'),
        'cause': details.get('cause', 'No information available'),
        'medicine_image': details.get('medicine_image', '/static/default_medicine.png')
    })

if __name__ == '__main__':
    app.run(debug=True)
