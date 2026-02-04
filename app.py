from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

import onnxruntime as ort

# Load model
model_session = None

def load_model():
    global model_session
    model_path = 'model.onnx'
    if os.path.exists(model_path):
        model_session = ort.InferenceSession(model_path)
        print(f'Model loaded from {model_path}')
    else:
        print(f'Warning: Model not found at {model_path}')

@app.route('/')
def index():
    return render_template('web_app.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_session is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Load and preprocess image
        image = Image.open(file.stream).convert('RGB')
        original_size = image.size
        
        # Resize to model input size (128x128) matches export script
        image_resized = image.resize((128, 128), Image.BILINEAR)
        
        # Convert to numpy array and normalize
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        
        # Transpose to (C, H, W) and add batch dimension -> (1, 3, 128, 128)
        input_tensor = np.transpose(image_array, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Predict using ONNX Runtime
        input_name = model_session.get_inputs()[0].name
        output_name = model_session.get_outputs()[0].name
        
        outputs = model_session.run([output_name], {input_name: input_tensor})
        probability_map = outputs[0][0, 0] # Remove batch and channel dims
        
        # Resize back to original size
        prob_img = Image.fromarray((probability_map * 255).astype(np.uint8), mode='L')
        prob_img_resized = prob_img.resize(original_size, Image.BILINEAR)
        probability_map_resized = np.array(prob_img_resized).astype(np.float32) / 255.0
        
        # Calculate statistics
        max_prob = float(np.max(probability_map_resized))
        mean_prob = float(np.mean(probability_map_resized))
        min_prob = float(np.min(probability_map_resized))
        std_prob = float(np.std(probability_map_resized))
        median_prob = float(np.median(probability_map_resized))
        q25_prob = float(np.percentile(probability_map_resized, 25))
        q75_prob = float(np.percentile(probability_map_resized, 75))
        binary_mask = (probability_map_resized > 0.5).astype(np.float32)
        landslide_pixels = np.sum(binary_mask)
        total_pixels = binary_mask.size
        landslide_percentage = (landslide_pixels / total_pixels) * 100
        high_risk_pixels_70 = float(np.sum(probability_map_resized > 0.7))
        high_risk_pixels_90 = float(np.sum(probability_map_resized > 0.9))
        high_risk_percentage_70 = (high_risk_pixels_70 / total_pixels) * 100
        high_risk_percentage_90 = (high_risk_pixels_90 / total_pixels) * 100
        
        # Simple textual risk level based on landslide area
        if landslide_percentage < 5:
            risk_level = 'Çok Düşük'
        elif landslide_percentage < 15:
            risk_level = 'Düşük'
        elif landslide_percentage < 30:
            risk_level = 'Orta'
        elif landslide_percentage < 50:
            risk_level = 'Yüksek'
        else:
            risk_level = 'Çok Yüksek'
        
        # Convert probability map to base64
        prob_img_bytes = io.BytesIO()
        prob_img_resized.save(prob_img_bytes, format='PNG')
        prob_img_base64 = base64.b64encode(prob_img_bytes.getvalue()).decode('utf-8')
        
        return jsonify({
            'probability': float(mean_prob),
            'max_prob': float(max_prob),
            'mean_prob': float(mean_prob),
            'min_prob': float(min_prob),
            'std_prob': float(std_prob),
            'median_prob': float(median_prob),
            'q25_prob': float(q25_prob),
            'q75_prob': float(q75_prob),
            'landslide_percentage': float(landslide_percentage),
            'high_risk_percentage_70': float(high_risk_percentage_70),
            'high_risk_percentage_90': float(high_risk_percentage_90),
            'risk_level': risk_level,
            'probability_map': prob_img_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    print('Starting web server...')
    print('Open http://localhost:5000 in your browser')
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

