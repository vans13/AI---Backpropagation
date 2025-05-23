=========================================
README - Detector de Objetos con YOLOv8
=========================================

Este proyecto utiliza un modelo YOLOv8 pre-entrenado para realizar detección de objetos en imágenes y videos. 
El script principal es `PrediccionYolo.py`.

-------------------------
1. PROPÓSITO DEL SCRIPT
-------------------------
El script `PrediccionYolo.py` permite al usuario seleccionar un archivo de imagen o video y 
realiza la detección de objetos utilizando el modelo YOLOv8 especificado en el código.
Muestra los resultados visualmente y en la consola.

-------------------------
2. REQUISITOS
-------------------------
- Python 3.8 o superior.
- Las siguientes librerías de Python (se pueden instalar con pip):
  - ultralytics (para YOLOv8)
  - opencv-python (cv2)
  - numpy
  - torch (PyTorch, debe instalarse con soporte CUDA para aceleración por GPU)
  - tkinter (generalmente incluido con Python, usado para el diálogo de selección de archivo)

- Para aceleración por GPU (altamente recomendado para modelos grandes y video):
  - Una GPU NVIDIA compatible con CUDA.
  - Drivers NVIDIA actualizados.
  - CUDA Toolkit (la versión debe ser compatible con la versión de PyTorch instalada).

- Archivos de modelo YOLO (ej. `yolov8x.pt`):
  El script está configurado para usar un nombre de modelo específico (MODEL_NAME). 
  La librería `ultralytics` descargará automáticamente el modelo si no se encuentra localmente 
  en la primera ejecución.

-------------------------
3. CÓMO EJECUTAR EL SCRIPT
-------------------------
1. Asegúrate de tener todos los requisitos instalados.
2. Abre una terminal o línea de comandos.
3. Navega hasta el directorio donde se encuentra `PrediccionYolo.py`.
4. Ejecuta el script con el comando:
   python PrediccionYolo.py
5. Se abrirá una ventana de diálogo para que selecciones un archivo de imagen o video.
   Formatos de imagen comunes: .jpg, .jpeg, .png, .bmp
   Formatos de video comunes: .mp4, .avi, .mov, .mkv

-------------------------
4. FUNCIONALIDADES
-------------------------
- **Diagnóstico del Entorno:** Al inicio, imprime información sobre la versión de PyTorch y la disponibilidad de CUDA.
- **Selección de Archivo:** Permite al usuario elegir interactivamente el archivo a procesar.
- **Detección de Tipo de Archivo:** Identifica automáticamente si el archivo es una imagen o un video por su extensión.
- **Modelo YOLOv8:** Utiliza el modelo definido en la variable `MODEL_NAME` (por defecto `yolov8x.pt`) para la detección.
- **Detección en Imágenes:**
  - Muestra la imagen en una ventana.
  - Al pasar el cursor del mouse sobre las áreas donde se detectaron objetos, se mostrará un cuadro delimitador y una etiqueta con la clase del objeto y la confianza.
  - Al cerrar la ventana, se imprimen en consola los detalles de todas las detecciones.
- **Detección en Videos:**
  - Muestra el video con las detecciones dibujadas en tiempo real.
  - La velocidad de reproducción se ajusta a los FPS originales del video.
  - Imprime en consola un resumen de las detecciones para el primer frame y luego periódicamente (aprox. cada segundo).
  - Al finalizar el video (o al interrumpirlo), imprime un resumen final en consola con el conteo total de cada clase de objeto detectado a lo largo de todo el video y su confianza promedio.
- **Salida por Consola Mejorada:** Proporciona información estructurada y clara sobre el proceso y los resultados.
- **Opciones para Salir:** Se puede salir de la visualización presionando 'q', 's', o cerrando la ventana con el botón 'X'.
- **Redimensionamiento para Visualización:** Las imágenes y videos se redimensionan si son muy grandes para ajustarse mejor a la pantalla, manteniendo la relación de aspecto.

-------------------------
5. PERSONALIZACIÓN (Opcional)
-------------------------
- **Cambiar el Modelo YOLO:**
  Puedes cambiar el modelo YOLOv8 utilizado modificando la variable `MODEL_NAME` cerca del inicio del script `PrediccionYolo.py`.
  Ejemplos:
    MODEL_NAME = 'yolov8n.pt'  # Modelo Nano (rápido, menos preciso)
    MODEL_NAME = 'yolov8s.pt'  # Modelo Pequeño
    MODEL_NAME = 'yolov8m.pt'  # Modelo Mediano
    MODEL_NAME = 'yolov8l.pt'  # Modelo Grande
    MODEL_NAME = 'yolov8x.pt'  # Modelo Extra Grande (lento, más preciso)
  El modelo se descargará automáticamente si es la primera vez que se usa.

- **Ajustar Parámetros de Visualización:**
  Las constantes `MAX_DISPLAY_WIDTH` y `MAX_DISPLAY_HEIGHT` se pueden modificar para cambiar el tamaño máximo de la ventana de visualización.

-------------------------
6. NOTAS ADICIONALES
-------------------------
- El rendimiento, especialmente en videos y con modelos grandes, depende en gran medida de si se está utilizando una GPU con CUDA.
- La precisión de la detección depende del modelo YOLO utilizado y de la calidad/claridad del archivo de entrada.