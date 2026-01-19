"""
API Flask para reconocimiento de imágenes con TensorFlow Lite.
"""

from flask import Flask, request, jsonify
import os
from .image_utils import load_image_from_url, load_image_from_file, preprocess_image
from .model_loader import load_model, predict, is_model_loaded, get_model_info
import json

app = Flask(__name__)

# Configuración
MODEL_PATH = os.getenv('MODEL_PATH', 'plant_species.tflite')
DEFAULT_TARGET_SIZE = (256, 256)  # Puede ajustarse según el modelo

# Cargar el modelo al iniciar la aplicación
try:
    load_model(MODEL_PATH)
    print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    print("La aplicación puede no funcionar correctamente.")


def predict_class_name(class_idx):
    """
    Lee el archivo labels.json y retorna el nombre de la clase dado un índice.
    labels.json tiene el formato: {"0": ..., "1": ..., ...}
    """
    try:
        # Buscar labels.json en la misma carpeta que app.py
        labels_path = os.path.join(os.path.dirname(__file__), 'labels.json')
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        clase_nombre = labels[str(class_idx)]
        return clase_nombre
    except Exception:
        return f"Clase {class_idx}"


@app.route('/predict', methods=['GET'])
def predict_page():
    """Página HTML interactiva para realizar predicciones sobre imágenes."""
    html = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reconocimiento de Plantas - Clasificador</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }
            
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2em;
            }
            
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            
            .upload-section {
                margin-bottom: 30px;
            }
            
            .button-group {
                display: flex;
                gap: 15px;
                justify-content: center;
                flex-wrap: wrap;
                margin-bottom: 20px;
            }
            
            .btn {
                padding: 15px 30px;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .btn-primary {
                background: #667eea;
                color: white;
            }
            
            .btn-primary:hover {
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            .btn-secondary {
                background: #48bb78;
                color: white;
            }
            
            .btn-secondary:hover {
                background: #38a169;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(72, 187, 120, 0.4);
            }
            
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            input[type="file"] {
                display: none;
            }
            
            .preview-section {
                text-align: center;
                margin: 30px 0;
            }
            
            .preview-image {
                max-width: 100%;
                max-height: 400px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                display: none;
                margin: 0 auto;
            }
            
            .preview-image.show {
                display: block;
            }
            
            .camera-preview {
                max-width: 100%;
                max-height: 400px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                display: none;
                margin: 0 auto;
            }
            
            .camera-preview.show {
                display: block;
            }
            
            .camera-controls {
                text-align: center;
                margin-top: 15px;
                display: none;
            }
            
            .camera-controls.show {
                display: block;
            }
            
            .result-section {
                margin-top: 30px;
                padding: 20px;
                background: #f7fafc;
                border-radius: 15px;
                display: none;
            }
            
            .result-section.show {
                display: block;
            }
            
            .result-success {
                background: #f0fff4;
                border-left: 4px solid #48bb78;
            }
            
            .result-error {
                background: #fff5f5;
                border-left: 4px solid #f56565;
            }
            
            .result-title {
                font-size: 1.5em;
                font-weight: bold;
                margin-bottom: 15px;
                color: #333;
            }
            
            .result-class {
                font-size: 1.8em;
                color: #667eea;
                font-weight: bold;
                margin: 10px 0;
            }
            
            .result-confidence {
                font-size: 1.2em;
                color: #666;
                margin: 10px 0;
            }
            
            .loading {
                text-align: center;
                padding: 20px;
                display: none;
            }
            
            .loading.show {
                display: block;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error-message {
                color: #f56565;
                font-weight: bold;
            }
            
            @media (max-width: 600px) {
                .container {
                    padding: 20px;
                }
                
                h1 {
                    font-size: 1.5em;
                }
                
                .button-group {
                    flex-direction: column;
                }
                
                .btn {
                    width: 100%;
                    justify-content: center;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌿 Reconocimiento de Plantas</h1>
            <p class="subtitle">Sube una imagen o usa tu cámara para identificar la especie</p>
            
            <div class="upload-section">
                <div class="button-group">
                    <button class="btn btn-secondary" onclick="openCamera()">
                        📷 Usar Cámara
                    </button>
                    <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">
                        📁 Seleccionar Imagen
                    </button>
                    <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
                </div>
                
                <div class="preview-section">
                    <img id="previewImage" class="preview-image" alt="Preview">
                    <video id="cameraPreview" class="camera-preview" autoplay playsinline></video>
                    <div class="camera-controls">
                        <button class="btn btn-primary" onclick="capturePhoto()">📸 Capturar Foto</button>
                        <button class="btn btn-secondary" onclick="stopCamera()">❌ Cerrar Cámara</button>
                    </div>
                </div>
                
                <div class="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 15px; color: #666;">Procesando imagen...</p>
                </div>
                
                <div id="resultSection" class="result-section">
                    <div id="resultContent"></div>
                </div>
            </div>
        </div>
        
        <script>
            let stream = null;
            const fileInput = document.getElementById('fileInput');
            const previewImage = document.getElementById('previewImage');
            const cameraPreview = document.getElementById('cameraPreview');
            const cameraControls = document.querySelector('.camera-controls');
            const resultSection = document.getElementById('resultSection');
            const resultContent = document.getElementById('resultContent');
            const loading = document.querySelector('.loading');
            let capturedBlob = null;
            
            function openCamera() {
                navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
                    .then(mediaStream => {
                        stream = mediaStream;
                        cameraPreview.srcObject = stream;
                        cameraPreview.classList.add('show');
                        cameraControls.classList.add('show');
                        previewImage.classList.remove('show');
                        resultSection.classList.remove('show');
                    })
                    .catch(err => {
                        alert('Error al acceder a la cámara: ' + err.message);
                    });
            }
            
            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }
                cameraPreview.classList.remove('show');
                cameraControls.classList.remove('show');
            }
            
            function capturePhoto() {
                const canvas = document.createElement('canvas');
                canvas.width = cameraPreview.videoWidth;
                canvas.height = cameraPreview.videoHeight;
                canvas.getContext('2d').drawImage(cameraPreview, 0, 0);
                
                canvas.toBlob(blob => {
                    capturedBlob = blob;
                    previewImage.src = URL.createObjectURL(blob);
                    previewImage.classList.add('show');
                    stopCamera();
                    uploadImage(blob);
                }, 'image/jpeg');
            }
            
            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (file) {
                    previewImage.src = URL.createObjectURL(file);
                    previewImage.classList.add('show');
                    resultSection.classList.remove('show');
                    uploadImage(file);
                }
            }
            
            function uploadImage(imageBlob) {
                const formData = new FormData();
                formData.append('image_file', imageBlob, 'image.jpg');
                
                loading.classList.add('show');
                resultSection.classList.remove('show');
                
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    loading.classList.remove('show');
                    displayResult(data);
                })
                .catch(error => {
                    loading.classList.remove('show');
                    displayError('Error al procesar la imagen: ' + error.message);
                });
            }
            
            function displayResult(data) {
                resultSection.classList.add('show');
                
                if (data.success) {
                    resultSection.className = 'result-section show result-success';
                    resultContent.innerHTML = `
                        <div class="result-title">✅ Resultado de la Predicción</div>
                        <div class="result-class">${data.class}</div>
                        <div class="result-confidence">Confianza: ${data.confidence}</div>
                    `;
                } else {
                    resultSection.className = 'result-section show result-error';
                    resultContent.innerHTML = `
                        <div class="result-title">❌ Error</div>
                        <div class="error-message">${data.error || 'Error desconocido'}</div>
                    `;
                }
            }
            
            function displayError(message) {
                resultSection.classList.add('show');
                resultSection.className = 'result-section show result-error';
                resultContent.innerHTML = `
                    <div class="result-title">❌ Error</div>
                    <div class="error-message">${message}</div>
                `;
            }
        </script>
    </body>
    </html>
    """
    return html


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Endpoint POST para procesar imágenes y realizar predicciones.
    
    Acepta:
    - image_file: archivo de imagen (multipart/form-data)
    
    Returns:
        JSON con class, confidence y success
    """
    try:
        image_array = None
        
        # Intentar obtener imagen desde archivo
        if 'image_file' in request.files:
            file = request.files['image_file']
            if file.filename != '':
                image_array = load_image_from_file(file)
        
        # Si no hay archivo, intentar obtener desde URL
        if image_array is None:
            if request.is_json:
                data = request.get_json()
                image_url = data.get('image_url')
            else:
                image_url = request.form.get('image_url')
            
            if image_url:
                image_array = load_image_from_url(image_url)
        
        # Validar que se obtuvo una imagen
        if image_array is None:
            return jsonify({
                'success': False,
                'error': 'No se proporcionó imagen. Use image_file o image_url.'
            }), 400
        
        # Preprocesar la imagen
        processed_image = preprocess_image(image_array, target_size=DEFAULT_TARGET_SIZE)
        
        # Realizar predicción
        class_idx, confidence = predict(processed_image)
        confidence = confidence * 100

        class_name = predict_class_name(class_idx)

        # Retornar resultado
        return jsonify({
            'success': True,
            'class': class_name,
            'confidence': f"{confidence:.3f}%"
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except IOError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error interno del servidor: {str(e)}'
        }), 500


@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home():
    """Endpoint de validación visible desde el navegador."""
    model_loaded = is_model_loaded()
    model_status = "Cargado" if model_loaded else "No cargado"
    
    model_info = get_model_info()
    if model_info:
        model_path = model_info['model_path']
        input_shape = model_info['input_shape']
        if input_shape:
            input_size_display = f"{input_shape[1]}x{input_shape[0]}" if len(input_shape) >= 2 else "N/A"
        else:
            input_size_display = f"{DEFAULT_TARGET_SIZE[0]}x{DEFAULT_TARGET_SIZE[1]}"
    else:
        model_path = MODEL_PATH
        input_size_display = f"{DEFAULT_TARGET_SIZE[0]}x{DEFAULT_TARGET_SIZE[1]}"
    
    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>API de Reconocimiento de Imágenes - Estado</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }}
            .container {{
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 600px;
                width: 90%;
            }}
            h1 {{
                color: #333;
                text-align: center;
                margin-bottom: 10px;
            }}
            .status {{
                text-align: center;
                margin: 30px 0;
            }}
            .status-badge {{
                display: inline-block;
                padding: 10px 30px;
                border-radius: 25px;
                font-size: 18px;
                font-weight: bold;
                color: white;
                background: #4CAF50;
                box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
            }}
            .info {{
                background: #f5f5f5;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
            }}
            .info-item {{
                margin: 15px 0;
                padding: 10px;
                border-left: 4px solid #667eea;
                background: white;
                border-radius: 5px;
            }}
            .info-label {{
                font-weight: bold;
                color: #555;
                margin-bottom: 5px;
            }}
            .info-value {{
                color: #333;
                font-family: 'Courier New', monospace;
            }}
            .endpoints {{
                margin-top: 30px;
                padding-top: 20px;
                border-top: 2px solid #eee;
            }}
            .endpoint {{
                background: #e3f2fd;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #2196F3;
            }}
            .endpoint-method {{
                display: inline-block;
                background: #2196F3;
                color: white;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                margin-right: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌿 API de Reconocimiento de Plantas</h1>
            <div class="status">
                <div class="status-badge">✓ API ACTIVA</div>
            </div>
            <div class="info">
                <div class="info-item">
                    <div class="info-label">Estado del Modelo:</div>
                    <div class="info-value">{model_status}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Ruta del Modelo:</div>
                    <div class="info-value">{model_path}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Tamaño de Entrada:</div>
                    <div class="info-value">{input_size_display} píxeles</div>
                </div>
            </div>
            <div class="endpoints">
                <h3 style="color: #333; margin-top: 0;">Endpoints Disponibles:</h3>
                <div class="endpoint">
                    <span class="endpoint-method">GET</span>
                    <strong>/home</strong> - Página de estado (esta página)
                </div>
                <div class="endpoint">
                    <span class="endpoint-method">GET</span>
                    <strong>/predict</strong> - Página de predicción (HTML)
                </div>
                <div class="endpoint">
                    <span class="endpoint-method" style="background: #4CAF50;">POST</span>
                    <strong>/predict</strong> - Predicción de imágenes (tflite)
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
