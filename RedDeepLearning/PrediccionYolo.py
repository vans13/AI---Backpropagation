# Importar las librerías necesarias
from ultralytics import YOLO
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
# from collections import Counter # No se necesita para el formato final elegido

# --- Constantes para la visualización ---
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720
HOVER_WINDOW_NAME = "Detecciones YOLOv8 - Imagen Interactiva"

# --- Variables globales para la función de callback del mouse (solo para imagen) ---
original_image_for_hover = None
detections_for_hover = []

# --- 1. Diagnóstico Inicial y Carga del Modelo ---
print("=" * 60)
print(" DIAGNÓSTICO DEL ENTORNO Y CARGA DEL MODELO DE IA")
print("=" * 60)
MODEL_NAME = 'yolov8x.pt'
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {is_cuda_available}")
    if is_cuda_available:
        print(f"CUDA version (PyTorch compiled with): {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU ID: {torch.cuda.current_device()}")
        print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("¡ADVERTENCIA!: CUDA no está disponible. PyTorch está usando la CPU.")
        print("   El rendimiento será significativamente menor. Revisa la instalación de PyTorch y CUDA.")
    print("-" * 60)
    
    model = YOLO(MODEL_NAME)
    print(f"Modelo IA ({MODEL_NAME}) cargado exitosamente.")
except Exception as e:
    print(f"Error crítico durante la inicialización o carga del modelo {MODEL_NAME}: {e}")
    print("El programa terminará.")
    exit()
print("=" * 60)


def redimensionar_para_mostrar(imagen):
    """Redimensiona una imagen para que se ajuste a las dimensiones máximas de visualización."""
    h, w = imagen.shape[:2]
    escala = 1.0
    if w > MAX_DISPLAY_WIDTH:
        escala = MAX_DISPLAY_WIDTH / w
    if h * escala > MAX_DISPLAY_HEIGHT:
        escala = MAX_DISPLAY_HEIGHT / h
    if escala < 1.0:
        nuevo_ancho = int(w * escala)
        nueva_altura = int(h * escala)
        imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nueva_altura), interpolation=cv2.INTER_AREA)
        return imagen_redimensionada
    return imagen

def mouse_event_detector_imagen(event, x, y, flags, param):
    """Callback para eventos del mouse; dibuja detecciones al pasar el cursor sobre ellas."""
    global original_image_for_hover, detections_for_hover, HOVER_WINDOW_NAME
    if event == cv2.EVENT_MOUSEMOVE and original_image_for_hover is not None:
        img_display_copy = original_image_for_hover.copy()
        for det in detections_for_hover:
            x1_b, y1_b, x2_b, y2_b = det['box']
            if x1_b < x < x2_b and y1_b < y < y2_b:
                label = f"{det['name']} {det['conf']:.2f}"
                cv2.rectangle(img_display_copy, (int(x1_b), int(y1_b)), (int(x2_b), int(y2_b)), (0, 255, 0), 2)
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = int(x1_b)
                text_y = int(y1_b) - 10
                if text_y < 0: text_y = int(y1_b) + text_size[1] + 5
                cv2.putText(img_display_copy, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                break
        cv2.imshow(HOVER_WINDOW_NAME, img_display_copy)

def detectar_en_imagen(ruta_imagen):
    """Procesa una imagen para detectar objetos y muestra los resultados interactivamente."""
    global original_image_for_hover, detections_for_hover, HOVER_WINDOW_NAME
    print(f"\nIniciando procesamiento de IMAGEN: {os.path.basename(ruta_imagen)}")
    print(f"Usando modelo: {MODEL_NAME}")
    print("-" * 60)
    try:
        img_cv2_original = cv2.imread(ruta_imagen)
        if img_cv2_original is None:
            print(f"Error: No se pudo cargar la imagen de {ruta_imagen}")
            return

        results = model(img_cv2_original, verbose=False)
        result = results[0]

        detections_for_hover.clear()
        if result.boxes:
            for box_data in result.boxes:
                x1, y1, x2, y2 = box_data.xyxy[0].cpu().numpy()
                conf = box_data.conf[0].item()
                cls_id = int(box_data.cls[0].item())
                class_name = model.names[cls_id]
                detections_for_hover.append({'box': [x1, y1, x2, y2], 'name': class_name, 'conf': conf})

        original_image_for_hover = redimensionar_para_mostrar(img_cv2_original.copy())
        h_orig, w_orig = img_cv2_original.shape[:2]
        h_disp, w_disp = original_image_for_hover.shape[:2]
        scale_x = w_disp / w_orig if w_orig > 0 else 1.0
        scale_y = h_disp / h_orig if h_orig > 0 else 1.0

        temp_detections_scaled = []
        for det in detections_for_hover:
            x1_orig, y1_orig, x2_orig, y2_orig = det['box']
            temp_detections_scaled.append({
                'box': [x1_orig * scale_x, y1_orig * scale_y, x2_orig * scale_x, y2_orig * scale_y],
                'name': det['name'], 'conf': det['conf']})
        detections_for_hover = temp_detections_scaled

        cv2.namedWindow(HOVER_WINDOW_NAME)
        cv2.setMouseCallback(HOVER_WINDOW_NAME, mouse_event_detector_imagen)
        cv2.imshow(HOVER_WINDOW_NAME, original_image_for_hover)
        print("INFO: Visualización interactiva iniciada.")
        print("      Mueve el cursor sobre objetos para ver detalles.")
        print("      Presiona 'q', 's' o cierra la ventana para terminar.")
        print("-" * 60)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('s'):
                print("INFO: Cierre solicitado por teclado.")
                break
            try:
                if cv2.getWindowProperty(HOVER_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("INFO: Ventana cerrada por el usuario.")
                    break
            except cv2.error:
                print("INFO: Ventana ya no existe (error al verificar propiedad).")
                break
        
        cv2.destroyAllWindows()
        original_image_for_hover = None
        detections_for_hover.clear()

        print("\n" + "=" * 60)
        print(f" RESULTADOS DETALLADOS DE LA IMAGEN (Modelo: {MODEL_NAME})")
        print("=" * 60)
        if not result.boxes or len(result.boxes) == 0:
            print("  No se detectaron objetos en la imagen.")
        else:
            print(f"  Total de objetos detectados: {len(result.boxes)}")
            for i, box_data in enumerate(result.boxes):
                coords_orig = box_data.xyxy[0].cpu().numpy().tolist()
                clase_id = int(box_data.cls[0].item())
                nombre_clase = model.names[clase_id]
                confianza = box_data.conf[0].item()
                print(f"\n  --- Objeto Detectado #{i+1} ---")
                print(f"    Clase        : {nombre_clase} (ID: {clase_id})")
                formatted_coords = [f"{c:.2f}" for c in coords_orig]
                print(f"    Coordenadas  : [xmin, ymin, xmax, ymax] = [{', '.join(formatted_coords)}]")
                print(f"    Confianza    : {confianza:.4f}")
        print("=" * 60)
    except Exception as e:
        print(f"Error GRAVE durante la detección en imagen: {e}")
        import traceback
        traceback.print_exc()

def detectar_en_video(ruta_video):
    """Procesa un video para detectar objetos y muestra los resultados frame a frame."""
    print(f"\nIniciando procesamiento de VIDEO: {os.path.basename(ruta_video)}")
    print(f"Usando modelo: {MODEL_NAME}")
    video_window_name = f"Detecciones YOLOv8 ({MODEL_NAME}) - Video"
    try:
        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {ruta_video}")
            return

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        default_fps_para_imprimir = 25 # Usado si fps_video es 0, para la frecuencia de impresión
        if fps_video == 0 or fps_video is None:
            print("Advertencia: No se pudieron obtener los FPS del video. Usando 25 FPS por defecto para reproducción y frecuencia de impresión.")
            fps_video = default_fps_para_imprimir
        
        delay_entre_frames_ms = int(1000 / fps_video)
        # Frecuencia para imprimir en consola (ej. una vez por segundo)
        intervalo_impresion_frames = int(fps_video) if fps_video > 0 else default_fps_para_imprimir

        print(f"INFO: Reproduciendo a ~{fps_video:.2f} FPS (delay: {delay_entre_frames_ms} ms).")
        print(f"      Se imprimirán detecciones en consola para el frame 1 y luego cada ~{intervalo_impresion_frames} frames.")
        print("      Presiona 'q', 's' o cierra la ventana para terminar.")
        print("-" * 60)
        
        frame_count = 0
        cv2.namedWindow(video_window_name)

        while cap.isOpened():
            ret, frame_original = cap.read()
            if not ret:
                print("\nINFO: Fin del video o error al leer el fotograma.")
                break
            
            frame_count += 1
            results = model(frame_original, verbose=False)
            result = results[0]
            frame_con_detecciones = result.plot()
            frame_display = redimensionar_para_mostrar(frame_con_detecciones)
            cv2.imshow(video_window_name, frame_display)

            key = cv2.waitKey(delay_entre_frames_ms) & 0xFF 
            if key == ord('q') or key == ord('s'):
                print("INFO: Procesamiento de video interrumpido por el usuario.")
                break
            try:
                if cv2.getWindowProperty(video_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("INFO: Ventana de video cerrada por el usuario.")
                    break
            except cv2.error:
                print("INFO: Ventana de video ya no existe (error al verificar propiedad).")
                break
            
            # --- Nueva lógica para imprimir detecciones del video en consola ---
            # Imprimir para el primer frame y luego periódicamente (aprox. una vez por segundo)
            if frame_count == 1 or (intervalo_impresion_frames > 0 and frame_count % intervalo_impresion_frames == 0) :
                print("\n" + "-" * 40)
                print(f" DETECCIONES EN FRAME {frame_count} ")
                print("-" * 40)
                if result.boxes and len(result.boxes) > 0:
                    num_objetos = len(result.boxes)
                    lista_detallada_objetos = []
                    for box_data in result.boxes:
                        nombre_clase = model.names[int(box_data.cls[0].item())]
                        confianza = box_data.conf[0].item()
                        lista_detallada_objetos.append(f"{nombre_clase} ({confianza:.2f})")
                    
                    print(f"  Total de objetos: {num_objetos}")
                    print(f"  Lista: [{', '.join(lista_detallada_objetos)}]")
                else:
                    print(f"  No se detectaron objetos en este frame.")
                print("-" * 40)
        
        cap.release()
        cv2.destroyAllWindows()
        print("-" * 60)
        print("Procesamiento de video finalizado.")

    except Exception as e:
        print(f"Error GRAVE durante la detección en video: {e}")
        import traceback
        traceback.print_exc()

def elegir_archivo_y_tipo():
    """Abre un diálogo para seleccionar un archivo y determina si es imagen o video por su extensión."""
    root_tk = tk.Tk()
    root_tk.withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecciona un archivo de imagen o video",
        filetypes=(
            ("Archivos multimedia", "*.jpg *.jpeg *.png *.bmp *.gif *.mp4 *.avi *.mov *.mkv"),
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv"), ("Todos los archivos", "*.*")))
    root_tk.destroy()
    if not file_path: return None, None
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    if ext in image_extensions: return file_path, "imagen"
    elif ext in video_extensions: return file_path, "video"
    else:
        print(f"Advertencia: Extensión '{ext}' no reconocida como imagen/video compatible.")
        return file_path, "desconocido"

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" SELECCIÓN DE ARCHIVO MULTIMEDIA")
    print("=" * 60)
    ruta_archivo, tipo_archivo = elegir_archivo_y_tipo()

    if ruta_archivo:
        print(f"Archivo seleccionado: {ruta_archivo}")
        print(f"Tipo detectado: {tipo_archivo}")
        print("=" * 60)

        if tipo_archivo == "imagen":
            detectar_en_imagen(ruta_archivo)
        elif tipo_archivo == "video":
            detectar_en_video(ruta_archivo)
        elif tipo_archivo == "desconocido":
            print("Error: No se puede procesar. Tipo de archivo desconocido o no soportado.")
    else:
        print("INFO: No se seleccionó ningún archivo.")

    print("\n" + "=" * 60)
    print(" --- SCRIPT FINALIZADO ---")
    print("=" * 60)