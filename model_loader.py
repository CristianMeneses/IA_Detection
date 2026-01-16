"""
Cargador y manejador del modelo TensorFlow Lite para predicciones.
"""

import numpy as np
import tensorflow as tf
import threading
import os
import requests # Se añade para descargar el modelo


class ModelLoader:
    """Clase para cargar y usar modelos TensorFlow Lite."""
    
    def __init__(self, model_path):
        """
        Inicializa el cargador de modelo.
        
        Args:
            model_path (str): Ruta al archivo .tflite
        """
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        # Lock para evitar problemas en requests concurrentes
        self.interpreter_lock = threading.Lock()
        self._load_model()

    def _download_model_if_not_exists(self, model_url, local_path):
        """
        Descarga el modelo si no existe localmente.
        """
        if not os.path.exists(local_path):
            print(f"Descargando modelo desde {model_url} a {local_path}...")
            try:
                response = requests.get(model_url, stream=True, timeout=30)
                response.raise_for_status() # Lanza un error para códigos de estado HTTP malos
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Descarga completada.")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Error al descargar el modelo: {e}")
        else:
            print(f"El modelo ya existe localmente en {local_path}.")
    
    def _load_model(self):
        """Carga el modelo TensorFlow Lite."""
        try:
            # Define la URL de descarga del modelo
            # IMPORTANTE: REEMPLAZA ESTA URL CON LA URL DE DESCARGA DIRECTA DE TU MODELO
            MODEL_DOWNLOAD_URL = "https://huggingface.co/cmeneses99/IA_Detection/resolve/main/plant_species.tflite?download=true" 
            
            # Path local para guardar el modelo (usando /tmp/ en Replit es buena práctica)
            # Asegúrate de que el directorio /tmp exista, aunque en Replit suele estar disponible
            local_model_path = os.path.join("/tmp", os.path.basename(self.model_path))
            
            # Descargar el modelo si no existe
            self._download_model_if_not_exists(MODEL_DOWNLOAD_URL, local_model_path)

            self.interpreter = tf.lite.Interpreter(model_path=local_model_path)
            self.interpreter.allocate_tensors()
            
            # Obtener detalles de entrada y salida
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo: {str(e)}")
    
    def get_input_shape(self):
        """
        Obtiene la forma esperada de entrada del modelo.
        
        Returns:
            tuple: Forma de entrada (altura, ancho, canales)
        """
        if self.input_details:
            shape = self.input_details[0]['shape']
            # Retornar sin la dimensión de batch si existe
            if len(shape) == 4:
                return tuple(shape[1:4])  # (H, W, C)
            return tuple(shape)
        return None
    
    def predict(self, image_array):
        """
        Ejecuta una predicción sobre una imagen preprocesada.
        
        Args:
            image_array (np.ndarray): Array numpy de la imagen preprocesada
            
        Returns:
            tuple: (clase_predicha, confianza) donde confianza es un float entre 0 y 1
        """
        if self.interpreter is None:
            raise RuntimeError("Modelo no cargado. Llame a load_model() primero.")
        
        # Obtener la forma esperada de entrada
        input_shape = self.input_details[0]['shape']
        
        # Agregar dimensión de batch si es necesario
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Asegurar que el tipo de dato es correcto
        input_dtype = self.input_details[0]['dtype']
        if input_dtype == np.uint8:
            # Si el modelo espera uint8, convertir de float [0,1] a uint8 [0,255]
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
        else:
            image_array = image_array.astype(input_dtype)
        
        # Establecer el tensor de entrada y ejecutar inferencia con lock
        # para evitar problemas de concurrencia en Flask con m?ltiples requests
        with self.interpreter_lock:
            self.interpreter.set_tensor(self.input_details[0]['index'], image_array)
            
            # Ejecutar la inferencia
            self.interpreter.invoke()
            
            # Obtener las predicciones
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Procesar la salida (asumiendo que es un array de probabilidades)
        predictions = output_data[0]  # Remover dimensi?n de batch
        
        # Obtener la clase con mayor confianza
        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx])
        
        # Si las probabilidades no est?n normalizadas, normalizarlas
        if predictions.sum() > 1.1:  # Tolerancia para errores de punto flotante
            predictions = predictions / predictions.sum()
            confidence = float(predictions[class_idx])
        
        return int(class_idx), confidence
    

# Instancia global del modelo (se inicializará en app.py)
_model_instance = None


def load_model(model_path):
    """
    Carga el modelo TensorFlow Lite globalmente.
    
    Args:
        model_path (str): Ruta al archivo .tflite
    """
    global _model_instance
    _model_instance = ModelLoader(model_path)

def predict(image_array):
    """
    Ejecuta una predicción usando el modelo cargado globalmente.
    
    Args:
        image_array (np.ndarray): Array numpy de la imagen preprocesada
        
    Returns:
        tuple: (clase_predicha, confianza)
    """
    if _model_instance is None:
        raise RuntimeError("Modelo no cargado. Llame a load_model() primero.")
    
    return _model_instance.predict(image_array)

def is_model_loaded():
    """
    Verifica si el modelo está cargado.
    
    Returns:
        bool: True si el modelo está cargado, False en caso contrario
    """
    return _model_instance is not None


def get_model_info():
    """
    Obtiene información sobre el modelo cargado.
    
    Returns:
        dict: Diccionario con información del modelo o None si no está cargado
    """
    if _model_instance is None:
        return None
    
    return {
        'model_path': _model_instance.model_path,
        'input_shape': _model_instance.get_input_shape()
    }
