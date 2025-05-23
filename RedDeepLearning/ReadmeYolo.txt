==================================================================
  Aplicación de Detección de Objetos con YOLOv8 - UniCundinamarca
==================================================================

Este script de Python proporciona una interfaz gráfica de usuario (GUI) para realizar 
detección de objetos en imágenes y videos utilizando modelos YOLOv8 pre-entrenados.

-----------------------------
1. PROPÓSITO
-----------------------------
Facilitar la experimentación y visualización de las capacidades de los modelos YOLOv8 
para la detección de objetos en diversos tipos de medios visuales.

-----------------------------
2. REQUISITOS
-----------------------------
* Python 3.8 o superior.
* Librerías de Python (instalar con pip, ej: pip install ultralytics opencv-python Pillow):
    - ultralytics (para YOLOv8)
    - opencv-python (cv2)
    - Pillow (PIL) (para manejo de imágenes en Tkinter)
    - numpy (generalmente instalado como dependencia)
    - torch (PyTorch - para un rendimiento óptimo, instalar con soporte CUDA)
    - tkinter (incluido en la instalación estándar de Python)

* Para Aceleración por GPU (Recomendado):
    - GPU NVIDIA compatible con CUDA.
    - Drivers NVIDIA actualizados.
    - CUDA Toolkit (compatible con tu versión de PyTorch y drivers).

* Archivo de Logo (Opcional):
    - Un archivo llamado `logo_unicundi.png` en la misma carpeta que el script para mostrar el logo de la universidad.

-----------------------------
3. CÓMO EJECUTAR LA APLICACIÓN
-----------------------------
1. Asegúrate de cumplir con todos los requisitos listados arriba.
2. Abre una terminal o consola de comandos.
3. Navega hasta el directorio donde guardaste el script (ej., `PrediccionYolo.py`).
4. Ejecuta el script usando el comando:
   python PrediccionYolo.py 
   (O el nombre que le hayas dado al archivo .py)

   Al iniciar, la aplicación intentará cargar un modelo YOLOv8 por defecto (ej. yolov8n.pt). 
   Espera a que aparezca el mensaje "Modelo (...) cargado exitosamente" en la barra de estado o consola.

-----------------------------
4. DESCRIPCIÓN DE LA INTERFAZ
-----------------------------

La interfaz se divide principalmente en un panel izquierdo para visualización y un panel derecho para controles e información.

PANEL IZQUIERDO:
* Visualización del Medio: Aquí se mostrará la imagen cargada o los fotogramas del video procesado.
    - Para Imágenes: Al pasar el cursor sobre un objeto detectado, se mostrará un recuadro verde y una etiqueta con la clase y confianza.
    - Para Videos: Los fotogramas del video con las detecciones se reproducirán aquí.

PANEL DERECHO:
* Archivo de Entrada:
    - [Label para Ruta]: Muestra la ruta completa del archivo seleccionado. Se ajusta si la ruta es larga.
    - [Botón "Seleccionar"]: Abre un diálogo para que elijas un archivo de imagen o video.
* Información del Archivo:
    - Muestra el tipo (Imagen/Video), dimensiones y tamaño en KB del archivo cargado.
* Selección de Modelo YOLOv8:
    - [Menú Desplegable]: Permite elegir diferentes variantes del modelo YOLOv8 (n, s, m, l, x). Por defecto, se carga 'yolov8n.pt'.
    - [Botón "Cargar Modelo"]: Carga el modelo seleccionado en el menú desplegable. La librería `ultralytics` lo descargará si no existe localmente. Espera la confirmación en la barra de estado.
    - [Label "Activo:"]: Muestra el nombre del modelo YOLOv8 que está actualmente cargado y listo para usarse.
* Botón "Ejecutar Detección":
    - Inicia el proceso de detección de objetos en el archivo de imagen o video actualmente seleccionado, utilizando el modelo YOLOv8 cargado. Este botón se habilita una vez que un modelo se ha cargado exitosamente.
* Resultados de Detección:
    - Un área de texto que muestra:
        - Para Imágenes: Un resumen de los objetos detectados, su clase y confianza, después de presionar "Ejecutar Detección".
        - Para Videos: Información del frame actual (número de frame, objetos detectados y sus confianzas) mientras se procesa. Al finalizar el video, muestra un resumen total de todos los objetos detectados, cuántas veces apareció cada clase y su confianza promedio.

PANEL SUPERIOR:
* Título de la Aplicación.
* Logo de la Universidad (si `logo_unicundi.png` está presente).
* Nombres de los integrantes.

BARRA DE ESTADO (Inferior):
* Muestra mensajes sobre el estado actual de la aplicación (ej. "Listo", "Cargando modelo...", "Procesando...", etc.).

-----------------------------
5. FUNCIONAMIENTO
-----------------------------
1. **Carga Inicial del Modelo:** Al iniciar, la aplicación carga un modelo YOLOv8 por defecto (definido en `DEFAULT_MODEL_NAME`). Espera a que se complete.
2. **Seleccionar Modelo (Opcional):** Puedes elegir un modelo diferente del menú desplegable y presionar "Cargar Modelo". La aplicación descargará el modelo si es necesario y lo cargará. El botón "Ejecutar Detección" se activará cuando el modelo esté listo.
3. **Seleccionar Archivo:** Haz clic en "Seleccionar" para elegir un archivo de imagen o video. Su información se mostrará.
4. **Ejecutar Detección:** Haz clic en "Ejecutar Detección".
   - Si es una **imagen**, se procesará y los resultados aparecerán en el área de texto. Podrás pasar el cursor sobre la imagen en el panel izquierdo para ver las detecciones resaltadas.
   - Si es un **video**, comenzará a reproducirse en el panel izquierdo con las detecciones dibujadas en cada fotograma. El área de texto mostrará información del frame actual. Al finalizar, se mostrará un resumen completo del video.
5. **Cambiar de Modelo/Archivo:** Puedes cargar un nuevo modelo o seleccionar un nuevo archivo en cualquier momento. Si cambias el modelo, asegúrate de presionar "Cargar Modelo" antes de "Ejecutar Detección" para usar el nuevo modelo.

-----------------------------
6. PARA SALIR DE LA APLICACIÓN
-----------------------------
Simplemente cierra la ventana principal de la aplicación.

-----------------------------
7. NOTAS
-----------------------------
- La velocidad de procesamiento, especialmente para videos y modelos grandes (como `yolov8x.pt`), depende significativamente de si tu sistema está utilizando una GPU con CUDA correctamente configurada. Si PyTorch usa la CPU, el procesamiento será mucho más lento.
- La calidad de las detecciones (precisión, objetos encontrados) dependerá del modelo YOLOv8 elegido y de la claridad y características del video o imagen de entrada.
- Los modelos YOLOv8 pre-entrenados están entrenados en el dataset COCO, que contiene 80 clases comunes de objetos.