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
        binary_mask = (probability_map_resized > 0.5).astype(np.float32)
        landslide_pixels = np.sum(binary_mask)
        total_pixels = binary_mask.size
        landslide_percentage = (landslide_pixels / total_pixels) * 100
        
        # Convert probability map to base64
        prob_img_bytes = io.BytesIO()
        prob_img_resized.save(prob_img_bytes, format='PNG')
        prob_img_base64 = base64.b64encode(prob_img_bytes.getvalue()).decode('utf-8')
        
        return jsonify({
            'probability': float(mean_prob),
            'max_prob': float(max_prob),
            'mean_prob': float(mean_prob),
            'landslide_percentage': float(landslide_percentage),
            'probability_map': prob_img_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    print('Starting web server...')
    print('Open http://localhost:5000 in your browser')
    app.run(debug=True, host='0.0.0.0', port=5000)

