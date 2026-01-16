"""
Utilidades para cargar y preprocesar imágenes para el modelo TensorFlow Lite.
"""

import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.applications.efficientnet import preprocess_input


def load_image_from_url(url):
    """
    Descarga una imagen desde una URL y la convierte a un array numpy RGB.
    
    Args:
        url (str): URL de la imagen a descargar
        
    Returns:
        np.ndarray: Array numpy con la imagen en formato RGB (H, W, 3)
        
    Raises:
        ValueError: Si la URL es inválida o la imagen no se puede descargar
        IOError: Si la imagen no se puede abrir o procesar
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')
        
        return np.array(image)
    except requests.RequestException as e:
        raise ValueError(f"Error al descargar imagen desde URL: {str(e)}")
    except Exception as e:
        raise IOError(f"Error al procesar imagen desde URL: {str(e)}")


def load_image_from_file(file):
    """
    Lee un archivo de imagen y lo convierte a un array numpy RGB.
    
    Args:
        file: Objeto de archivo (como el de Flask request.files)
        
    Returns:
        np.ndarray: Array numpy con la imagen en formato RGB (H, W, 3)
        
    Raises:
        IOError: Si el archivo no se puede abrir o procesar
    """
    try:
        image = Image.open(file)
        image = image.convert('RGB')
        
        return np.array(image)
    except Exception as e:
        raise IOError(f"Error al procesar archivo de imagen: {str(e)}")


def preprocess_image(image_array, target_size=(256, 256), use_efficientnet_preprocess=True):
    """
    Preprocesa una imagen para el modelo TensorFlow Lite.
    IMPORTANTE: Usa el mismo preprocesamiento que EfficientNet si el modelo fue entrenado con EfficientNet.
    
    Args:
        image_array (np.ndarray): Array numpy de la imagen en formato RGB (H, W, 3)
        target_size (tuple): Tamaño objetivo (ancho, alto). Default: (256, 256)
        use_efficientnet_preprocess (bool): Si True, usa preprocess_input de EfficientNet. Default: True
        
    Returns:
        np.ndarray: Array numpy preprocesado listo para el modelo
    """
    # Convertir a PIL Image si es necesario
    if isinstance(image_array, np.ndarray):
        image = Image.fromarray(image_array)
    else:
        image = image_array
    
    # Asegurar que sea RGB
    image = image.convert('RGB')
    
    # Redimensionar la imagen
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image_array = np.array(image, dtype=np.float32)
    
    # IMPORTANTE: Usar el mismo preprocesamiento que EfficientNet
    # Esto normaliza usando media y desviación estándar de ImageNet
    if use_efficientnet_preprocess:
        image_array = preprocess_input(image_array)
    else:
        # Fallback: normalización simple [0, 1]
        image_array = image_array / 255.0
    
    # La dimensión de batch se agregará en model_loader si es necesario
    return image_array
