from flask import Flask, request, render_template, jsonify
import torch
import numpy as np
from PIL import Image
import io
import base64
from model import get_model
import os

app = Flask(__name__)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

def load_model():
    global model
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        model = get_model(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f'Model loaded from {checkpoint_path}')
    else:
        print(f'Warning: Checkpoint not found at {checkpoint_path}')

@app.route('/')
def index():
    return render_template('web_app.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
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
        
        # Resize to model input size
        image_resized = image.resize((128, 128), Image.BILINEAR)
        
        # Convert to tensor
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probability_map = output[0].cpu().squeeze().numpy()
        
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
    app.run(debug=True, host='0.0.0.0', port=5000)

