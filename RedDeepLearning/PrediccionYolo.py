import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Label, Button, Frame, Canvas, PhotoImage, PanedWindow, Text, Scrollbar, StringVar, OptionMenu
from PIL import Image, ImageTk
import os
import cv2 # Para procesar video y leer imágenes
from ultralytics import YOLO # Para el modelo YOLO
import threading # Para evitar que la GUI se congele durante la carga inicial del modelo

# --- Constantes y Globales ---
DEFAULT_MODEL_NAME = 'yolov8n.pt' 
MAX_CANVAS_WIDTH = 620 
MAX_CANVAS_HEIGHT = 480 # Estos son más como valores iniciales o mínimos si el canvas no tiene tamaño aún

model = None 
current_loaded_model_name = "" 

current_media_path = None
current_media_type = None 
original_image_cv_rgb = None 
tk_image_on_canvas = None    
detections_for_hover = []    

video_capture = None
is_streaming_video_active = False
video_frame_count = 0
video_delay_ms = 30 # Se recalculará basado en FPS
acumulador_objetos_video_global = {}

canvas = None
file_path_display_var = None 
media_info_var = None
prediction_text_widget = None
status_bar_var = None
model_status_var = None 
selected_model_option_var = None
ejecutar_button = None 
load_model_button = None

# --- Funciones del Modelo y Procesamiento ---
def do_load_yolo_model_threaded(model_name_to_load):
    global status_bar_var, load_model_button, ejecutar_button
    if status_bar_var: status_bar_var.set(f"Iniciando carga de modelo: {model_name_to_load}...")
    if load_model_button: load_model_button.config(state=tk.DISABLED)
    if ejecutar_button: ejecutar_button.config(state=tk.DISABLED)
    window.update_idletasks()

    thread = threading.Thread(target=do_load_yolo_model_blocking, args=(model_name_to_load,))
    thread.daemon = True 
    thread.start()

def do_load_yolo_model_blocking(model_name_to_load):
    global model, current_loaded_model_name, status_bar_var, model_status_var, load_model_button, ejecutar_button
    
    print(f"Intentando cargar modelo: {model_name_to_load}")
    try:
        temp_model = YOLO(model_name_to_load) 
        
        # Verificación más robusta de que el modelo se cargó
        if hasattr(temp_model, 'predictor') and hasattr(temp_model, 'names') and len(temp_model.names) > 0:
            model = temp_model 
            current_loaded_model_name = model_name_to_load
            if status_bar_var: status_bar_var.set(f"Modelo ({current_loaded_model_name}) cargado exitosamente.")
            if model_status_var: model_status_var.set(f"Activo: {current_loaded_model_name}")
            print(f"Modelo ({current_loaded_model_name}) cargado. Info interna del modelo: {str(model.model.yaml)[:100]}...") # Imprimir parte de la config del modelo
            if ejecutar_button: ejecutar_button.config(state=tk.NORMAL)
        else:
            raise Exception("El modelo cargado no es válido o no tiene nombres de clases.")
            
    except Exception as e:
        if status_bar_var: status_bar_var.set(f"Error al cargar modelo {model_name_to_load}: {e}")
        if model_status_var: model_status_var.set(f"Error al cargar: {model_name_to_load}. Intente otro.")
        print(f"Error crítico al cargar el modelo {model_name_to_load}: {e}")
        model = None 
        if ejecutar_button: ejecutar_button.config(state=tk.DISABLED)
    finally:
        if load_model_button: load_model_button.config(state=tk.NORMAL)


def redimensionar_imagen_para_canvas(img_pil, target_width, target_height): # Ahora siempre usa target W/H
    w, h = img_pil.size
    if w == 0 or h == 0 or target_width <= 0 or target_height <= 0 : return None # Chequeos adicionales
    escala = min(target_width / w, target_height / h)
    
    # No escalar si la escala es 1 o mayor (ya cabe o es más pequeño), a menos que queramos forzar el llenado
    # Para llenar el espacio, siempre redimensionamos si la escala no es exactamente 1.
    # O, si queremos que llene y mantenga aspecto, la lógica actual de min(escala) está bien.
    # Si la imagen es más pequeña que el canvas, `escala` será > 1.
    # Para que llene el canvas pero sin distorsión (letterboxing/pillarboxing):
    nuevo_ancho = int(w * escala)
    nueva_altura = int(h * escala)
    
    return img_pil.resize((nuevo_ancho, nueva_altura), Image.Resampling.LANCZOS)

def procesar_imagen_seleccionada():
    global original_image_cv_rgb, detections_for_hover, prediction_text_widget, status_bar_var, model, current_loaded_model_name
    
    print(f"DEBUG: [procesar_imagen_seleccionada] Iniciando. Modelo actual en uso: {current_loaded_model_name}")
    if model and hasattr(model, 'model') and hasattr(model.model, 'yaml'):
        print(f"DEBUG: [procesar_imagen_seleccionada] Config del modelo (yaml): {str(model.model.yaml)[:100]}...")

    if original_image_cv_rgb is None:
        if status_bar_var: status_bar_var.set("Error: No hay imagen cargada.")
        return
    if model is None: 
        if status_bar_var: status_bar_var.set("Error: Modelo IA no cargado.")
        return

    if status_bar_var: status_bar_var.set("Procesando imagen con " + current_loaded_model_name + "...")
    window.update_idletasks()
    
    results = model(original_image_cv_rgb, verbose=False) # Usar el modelo global actual
    result = results[0]
    # detections_for_hover se limpia en ejecutar_cmd

    prediction_summary = f"Detecciones en Imagen ({current_loaded_model_name}):\n"
    if result.boxes and len(result.boxes) > 0:
        prediction_summary += f"  Total de objetos: {len(result.boxes)}\n"
        for i, box_data in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box_data.xyxy[0].cpu().numpy())
            conf = box_data.conf[0].item()
            cls_id = int(box_data.cls[0].item())
            nombre_clase = model.names[cls_id] 
            detections_for_hover.append({'box': (x1, y1, x2, y2), 'name': nombre_clase, 'conf': conf})
            prediction_summary += f"    {i+1}. {nombre_clase} (Conf: {conf:.2f})\n"
    else:
        prediction_summary += "  No se detectaron objetos."
    
    if prediction_text_widget:
        prediction_text_widget.config(state=tk.NORMAL)
        prediction_text_widget.delete(1.0, tk.END) 
        prediction_text_widget.insert(tk.END, prediction_summary)
        prediction_text_widget.config(state=tk.DISABLED)
    if status_bar_var: status_bar_var.set("Procesamiento completado. Mueva el cursor sobre la imagen.")

def _actualizar_frame_video():
    global video_capture, is_streaming_video_active, tk_image_on_canvas, canvas, model
    global prediction_text_widget, status_bar_var, acumulador_objetos_video_global, video_frame_count, video_delay_ms, current_loaded_model_name

    if not is_streaming_video_active or video_capture is None or model is None:
        return

    ret, frame = video_capture.read()
    if ret:
        video_frame_count += 1
        if video_frame_count == 1: # Imprimir info del modelo solo para el primer frame del video
            print(f"DEBUG: [Video Frame {video_frame_count}] Iniciando con modelo: {current_loaded_model_name}")
            if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
                 print(f"DEBUG: [Video Frame {video_frame_count}] Config del modelo (yaml): {str(model.model.yaml)[:100]}...")
        
        frame_rgb_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb_original, verbose=False, stream=False) 
        result = results[0]
        
        frame_para_plot = frame.copy() 
        plotted_frame_bgr = result.plot(img=frame_para_plot) 
        plotted_frame_rgb_pil = cv2.cvtColor(plotted_frame_bgr, cv2.COLOR_BGR2RGB)
        
        img_pil = Image.fromarray(plotted_frame_rgb_pil)
        
        canvas_w = canvas.winfo_width() if canvas.winfo_width() > 1 else MAX_CANVAS_WIDTH
        canvas_h = canvas.winfo_height() if canvas.winfo_height() > 1 else MAX_CANVAS_HEIGHT
        resized_pil = redimensionar_imagen_para_canvas(img_pil, canvas_w, canvas_h)

        if resized_pil:
            tk_image_on_canvas = ImageTk.PhotoImage(resized_pil)
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_image_on_canvas)
            canvas.image = tk_image_on_canvas 

            current_frame_summary = f"Frame: {video_frame_count} ({current_loaded_model_name})\n"
            if result.boxes and len(result.boxes) > 0:
                num_objetos = len(result.boxes)
                current_frame_summary += f"  Objetos detectados: {num_objetos}\n  Lista: ["
                det_list = []
                for box_data in result.boxes:
                    nombre_clase = model.names[int(box_data.cls[0].item())]
                    confianza = box_data.conf[0].item()
                    det_list.append(f"{nombre_clase} ({confianza:.2f})")
                    
                    if nombre_clase not in acumulador_objetos_video_global:
                        acumulador_objetos_video_global[nombre_clase] = {'total_confianza': 0.0, 'cantidad': 0}
                    acumulador_objetos_video_global[nombre_clase]['total_confianza'] += confianza
                    acumulador_objetos_video_global[nombre_clase]['cantidad'] += 1
                current_frame_summary += ", ".join(det_list) + "]"
            else:
                current_frame_summary += "  No se detectaron objetos."

            prediction_text_widget.config(state=tk.NORMAL)
            prediction_text_widget.delete(1.0, tk.END) 
            prediction_text_widget.insert(tk.END, current_frame_summary)
            prediction_text_widget.config(state=tk.DISABLED)

        window.after(video_delay_ms, _actualizar_frame_video)
    else:
        detener_stream_video()
        if status_bar_var: status_bar_var.set("Video finalizado o error al leer frame.")
        mostrar_resumen_final_video()

def detener_stream_video():
    global video_capture, is_streaming_video_active
    is_streaming_video_active = False
    if video_capture:
        video_capture.release()
        video_capture = None
    print("Video stream detenido.")

def mostrar_resumen_final_video():
    global acumulador_objetos_video_global, prediction_text_widget, current_media_path, current_loaded_model_name
    if prediction_text_widget is None: return

    video_name = os.path.basename(current_media_path) if current_media_path and current_media_type == "video" else "N/A"
    summary_text = f"Resumen Final del Video ({video_name} con {current_loaded_model_name}):\n"
    
    if not acumulador_objetos_video_global:
        summary_text += "  No se acumularon detecciones en este video."
    else:
        summary_text += "  Total de apariciones y Conf. Promedio:\n"
        objetos_ordenados = sorted(acumulador_objetos_video_global.items(), key=lambda item: item[1]['cantidad'], reverse=True)
        for nombre_clase, data in objetos_ordenados:
            cantidad = data['cantidad']
            conf_prom = data['total_confianza'] / cantidad if cantidad > 0 else 0
            summary_text += f"    - {nombre_clase:<20} | Visto: {cantidad:<5} | Conf. Prom.: {conf_prom:.2f}\n"
    
    prediction_text_widget.config(state=tk.NORMAL)
    prediction_text_widget.delete(1.0, tk.END) 
    prediction_text_widget.insert(tk.END, summary_text)
    prediction_text_widget.config(state=tk.DISABLED)

def iniciar_procesamiento_video(video_path):
    global video_capture, is_streaming_video_active, status_bar_var, video_delay_ms
    global acumulador_objetos_video_global, video_frame_count, canvas, model

    if model is None:
        messagebox.showerror("Error de Modelo", "El modelo IA no está cargado.")
        return

    detener_stream_video() 
    acumulador_objetos_video_global.clear()
    video_frame_count = 0
    
    if canvas: 
        canvas.delete("all")
        canvas.unbind("<Motion>")
        # No dibujar "Cargando video..." aquí, _actualizar_frame_video lo hará con el primer frame
        # Opcionalmente, un mensaje muy breve:
        canvas_w = canvas.winfo_width() if canvas.winfo_width() > 1 else MAX_CANVAS_WIDTH
        canvas_h = canvas.winfo_height() if canvas.winfo_height() > 1 else MAX_CANVAS_HEIGHT
        canvas.create_text(canvas_w/2, canvas_h/2, text="Iniciando video...", anchor="center", justify="center", font=("Arial",12))


    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        messagebox.showerror("Error", f"No se pudo abrir el video: {video_path}")
        return

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None: fps = 25 # Default FPS si no se puede leer
    video_delay_ms = int(1000 / fps) # Para reproducir a la velocidad original del video
    
    is_streaming_video_active = True
    if status_bar_var: status_bar_var.set(f"Reproduciendo video: {os.path.basename(video_path)} con {current_loaded_model_name}...")
    _actualizar_frame_video() 

def seleccionar_archivo_cmd():
    global current_media_path, current_media_type, original_image_cv_rgb, tk_image_on_canvas
    global canvas, file_path_display_var, media_info_var, prediction_text_widget, status_bar_var, detections_for_hover
    
    detener_stream_video() 

    file_path = filedialog.askopenfilename(
        title="Seleccionar archivo de Imagen o Video",
        filetypes=(("Archivos multimedia", "*.jpg *.jpeg *.png *.bmp *.gif *.mp4 *.avi *.mov *.mkv"),
                   ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif"),
                   ("Videos", "*.mp4 *.avi *.mov *.mkv")))
    if not file_path:
        if status_bar_var: status_bar_var.set("Selección de archivo cancelada.")
        return

    current_media_path = file_path
    file_path_display_var.set(file_path) 

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    media_info_text = ""
    detections_for_hover.clear() 
    if prediction_text_widget:
        prediction_text_widget.config(state=tk.NORMAL)
        prediction_text_widget.delete(1.0, tk.END)
        prediction_text_widget.config(state=tk.DISABLED)
    if canvas: canvas.delete("all")

    if ext in image_extensions:
        current_media_type = "imagen"
        try:
            img_cv2 = cv2.imread(current_media_path)
            original_image_cv_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(original_image_cv_rgb)
            
            canvas_w = canvas.winfo_width() if canvas.winfo_width() > 1 else MAX_CANVAS_WIDTH
            canvas_h = canvas.winfo_height() if canvas.winfo_height() > 1 else MAX_CANVAS_HEIGHT
            resized_pil = redimensionar_imagen_para_canvas(img_pil, canvas_w, canvas_h)
            
            if resized_pil:
                tk_image_on_canvas = ImageTk.PhotoImage(resized_pil)
                canvas.create_image(0, 0, anchor=tk.NW, image=tk_image_on_canvas)
                canvas.image = tk_image_on_canvas 
                canvas.bind("<Motion>", hover_sobre_imagen_cmd)
            else:
                canvas.create_text(canvas_w/2, canvas_h/2, text="Error al redimensionar imagen", anchor="center")

            h_orig, w_orig, c_orig = original_image_cv_rgb.shape
            size_kb = os.path.getsize(current_media_path) / 1024
            media_info_text = f"Tipo: Imagen\nDimensiones: {w_orig}x{h_orig}x{c_orig}\nTamaño: {size_kb:.2f} KB"
            if status_bar_var: status_bar_var.set("Imagen cargada. Presione 'Ejecutar' para detectar.")
        except Exception as e:
            print(f"Error al cargar imagen: {e}")
            media_info_text = "Error al cargar imagen."
            if status_bar_var: status_bar_var.set("Error al cargar imagen.")
            current_media_type = None; original_image_cv_rgb = None
    elif ext in video_extensions:
        current_media_type = "video"
        original_image_cv_rgb = None 
        tk_image_on_canvas = None
        if canvas: 
            canvas.delete("all")
            canvas_w = canvas.winfo_width() if canvas.winfo_width() > 1 else MAX_CANVAS_WIDTH
            canvas_h = canvas.winfo_height() if canvas.winfo_height() > 1 else MAX_CANVAS_HEIGHT
            canvas.create_text(canvas_w/2, canvas_h/2, text="Video seleccionado.\nPresione 'Ejecutar'.", anchor="center", justify="center", font=("Arial",12))
            canvas.unbind("<Motion>")
        try:
            temp_cap = cv2.VideoCapture(current_media_path)
            w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = temp_cap.get(cv2.CAP_PROP_FPS); temp_cap.release()
            size_kb = os.path.getsize(current_media_path) / 1024
            media_info_text = f"Tipo: Video\nDimensiones: {w}x{h}\nFPS: {fps:.2f}\nTamaño: {size_kb:.2f} KB"
            if status_bar_var: status_bar_var.set("Video cargado. Presione 'Ejecutar' para procesar.")
        except Exception as e:
            media_info_text = "Error al obtener info del video."; current_media_type = None
    else:
        current_media_type = "desconocido"; media_info_text = "Tipo de archivo no soportado."
    media_info_var.set(media_info_text)

def ejecutar_cmd():
    global current_media_type, current_media_path, status_bar_var, model
    global detections_for_hover, prediction_text_widget, canvas, tk_image_on_canvas

    # Detener cualquier video activo antes de una nueva ejecución
    detener_stream_video()

    # Limpiar explícitamente los resultados y detecciones de la ejecución anterior
    detections_for_hover.clear()
    if prediction_text_widget:
        prediction_text_widget.config(state=tk.NORMAL)
        prediction_text_widget.delete(1.0, tk.END)
        prediction_text_widget.config(state=tk.DISABLED)
    
    # Si es una imagen y ya hay una imagen en el canvas, la redibujamos (sin hover boxes)
    # para asegurar que esté limpia antes de la nueva detección.
    if current_media_type == "imagen" and tk_image_on_canvas and canvas:
        canvas.delete("all") # Limpiar todo lo que haya en el canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image_on_canvas) # Redibujar imagen base
        canvas.image = tk_image_on_canvas # Mantener referencia
    elif current_media_type == "video" and canvas: # Si es video, limpiar canvas y mostrar placeholder
        canvas.delete("all")
        canvas_w = canvas.winfo_width() if canvas.winfo_width() > 1 else MAX_CANVAS_WIDTH
        canvas_h = canvas.winfo_height() if canvas.winfo_height() > 1 else MAX_CANVAS_HEIGHT
        canvas.create_text(canvas_w/2, canvas_h/2, text="Video listo para ejecutar.\nPresione 'Ejecutar'.", anchor="center", justify="center", font=("Arial",12))


    if not current_media_path:
        messagebox.showwarning("Archivo no seleccionado", "Por favor, seleccione un archivo.")
        return
    if model is None:
        messagebox.showerror("Modelo no cargado", "Cargue un modelo IA primero.")
        return

    if current_media_type == "imagen":
        procesar_imagen_seleccionada()
    elif current_media_type == "video":
        iniciar_procesamiento_video(current_media_path)
    else:
        messagebox.showerror("Error", "Tipo de archivo no soportado.")

def hover_sobre_imagen_cmd(event):
    global tk_image_on_canvas, canvas, detections_for_hover, original_image_cv_rgb
    if tk_image_on_canvas is None or original_image_cv_rgb is None : 
        return
    
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image_on_canvas)

    if not detections_for_hover: 
        return

    x_mouse, y_mouse = event.x, event.y
    orig_h, orig_w = original_image_cv_rgb.shape[:2]
    # Usar el tamaño actual de la imagen mostrada en el canvas (tk_image_on_canvas)
    disp_w, disp_h = tk_image_on_canvas.width(), tk_image_on_canvas.height()


    if disp_w == 0 or disp_h == 0 or orig_w == 0 or orig_h == 0: return

    scale_w_orig_to_disp = disp_w / orig_w
    scale_h_orig_to_disp = disp_h / orig_h

    for det in detections_for_hover:
        x1_orig, y1_orig, x2_orig, y2_orig = det['box']
        conf = det['conf']; nombre_clase = det['name']
        
        x1_disp = int(x1_orig * scale_w_orig_to_disp); y1_disp = int(y1_orig * scale_h_orig_to_disp)
        x2_disp = int(x2_orig * scale_w_orig_to_disp); y2_disp = int(y2_orig * scale_h_orig_to_disp)

        if x1_disp < x_mouse < x2_disp and y1_disp < y_mouse < y2_disp:
            label = f"{nombre_clase} ({conf:.2f})"
            canvas.create_rectangle(x1_disp, y1_disp, x2_disp, y2_disp, outline="lime green", width=2)
            text_y_pos = y1_disp - 10
            if text_y_pos < 5: text_y_pos = y1_disp + 15
            canvas.create_text(x1_disp, text_y_pos, anchor=tk.SW, fill="lime green", font=("Arial", 10, "bold"), text=label)
            break

def cargar_modelo_seleccionado_cmd():
    model_to_load = selected_model_option_var.get()
    do_load_yolo_model_threaded(model_to_load)

# --- Configuración de la Interfaz Gráfica Principal ---
window = tk.Tk()
window.title("Detector de Objetos con YOLOv8 - UniCundinamarca")
window.geometry("1100x800")

# ... (El resto de la configuración de la GUI, desde top_frame hasta el final,
#      es igual que la versión anterior. Lo omito aquí por brevedad, pero debe estar en tu script.)

# --- Sección Superior: Título, Logo, Integrantes ---
top_frame = Frame(window)
top_frame.pack(side=tk.TOP, fill=tk.X, pady=(10,0)) 
title_label = Label(top_frame, text="Aplicación Laboratorio IA: Detección de Objetos con YOLOv8", font=("Arial", 18, "bold"))
title_label.pack(side=tk.LEFT, padx=20, pady=(0,5))
logo_tk_ref = None 
try:
    logo_pil = Image.open("logo_unicundi.png")
    logo_resized_pil = logo_pil.resize((70, 70), Image.Resampling.LANCZOS) 
    logo_tk_ref = ImageTk.PhotoImage(logo_resized_pil) 
    logo_label_widget = Label(top_frame, image=logo_tk_ref)
    logo_label_widget.image = logo_tk_ref 
    logo_label_widget.pack(side=tk.RIGHT, padx=20, pady=(0,5))
except Exception as e:
    print(f"Advertencia: Logo no cargado ('logo_unicundi.png'). {e}")

subtitle_frame = Frame(window)
subtitle_frame.pack(side=tk.TOP, fill=tk.X, pady=(0,10)) 
integrantes_label = Label(subtitle_frame, text="Presentado por: Juan Esteban Fuentes, Laura Estefania Latorre, Duván Santiago Matallana", font=("Arial", 10))
integrantes_label.pack()
uni_label = Label(subtitle_frame, text="Universidad de Cundinamarca", font=("Arial", 10, "italic"))
uni_label.pack()

main_paned_window = PanedWindow(window, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bd=2, sashwidth=8)
main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

left_panel = Frame(main_paned_window, bd=2, relief=tk.GROOVE)
main_paned_window.add(left_panel, stretch="always", width=650, minsize=400) # Darle más peso inicial al panel izquierdo
Label(left_panel, text="Visualización del Medio:", font=("Arial", 11, "bold")).pack(pady=(5,5), anchor="w", padx=10)
canvas = Canvas(left_panel, bg="lightgray", width=MAX_CANVAS_WIDTH, height=MAX_CANVAS_HEIGHT) 
canvas.pack(pady=5, padx=10, expand=True, fill=tk.BOTH) # expand y fill para que el canvas crezca

right_panel = Frame(main_paned_window, bd=2, relief=tk.GROOVE)
main_paned_window.add(right_panel, stretch="never", minsize=400) 

# Sección "Entrada"
entrada_frame = Frame(right_panel, pady=5)
entrada_frame.pack(fill=tk.X, padx=10, pady=(10,5))
Label(entrada_frame, text="Archivo de Entrada:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
file_path_display_var = StringVar()
file_path_label = Label(entrada_frame, textvariable=file_path_display_var, wraplength=320, justify=tk.LEFT, anchor="w", relief=tk.GROOVE, bd=1, padx=5)
file_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, pady=(2,5))
select_button = Button(entrada_frame, text="Seleccionar", command=seleccionar_archivo_cmd, width=10)
select_button.pack(side=tk.LEFT, padx=(5,0), pady=(2,5))

media_info_frame = Frame(right_panel, pady=5)
media_info_frame.pack(fill=tk.X, padx=10)
Label(media_info_frame, text="Información del Archivo:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
media_info_var = StringVar()
media_info_var.set("Tipo: N/A\nDimensiones: N/A\nTamaño: N/A")
info_label = Label(media_info_frame, textvariable=media_info_var, justify=tk.LEFT, anchor="w")
info_label.pack(anchor=tk.W, pady=(0,10))

modelo_frame = Frame(right_panel, pady=5)
modelo_frame.pack(fill=tk.X, padx=10)
Label(modelo_frame, text="Selección de Modelo YOLOv8:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
yolo_model_options = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
selected_model_option_var = StringVar(window)
selected_model_option_var.set(DEFAULT_MODEL_NAME)
model_option_menu = OptionMenu(modelo_frame, selected_model_option_var, *yolo_model_options)
model_option_menu.config(width=15)
model_option_menu.pack(side=tk.LEFT, pady=(2,5), fill=tk.X, expand=True)
load_model_button = Button(modelo_frame, text="Cargar Modelo", command=cargar_modelo_seleccionado_cmd, width=12)
load_model_button.pack(side=tk.LEFT, padx=(5,0), pady=(2,5))

model_status_var = StringVar()
model_status_var.set(f"Activo: (Cargando {DEFAULT_MODEL_NAME}...)")
model_status_label = Label(right_panel, textvariable=model_status_var, font=("Arial", 9, "italic"), wraplength=350)
model_status_label.pack(fill=tk.X, padx=10, pady=(0,10))

ejecutar_button = Button(right_panel, text="Ejecutar Detección", command=ejecutar_cmd, font=("Arial", 12, "bold"), bg="#AED6F1", state=tk.DISABLED, relief=tk.RAISED)
ejecutar_button.pack(pady=10, padx=10, fill=tk.X, ipady=5)

prediccion_frame = Frame(right_panel, pady=5)
prediccion_frame.pack(fill=tk.BOTH, expand=True, padx=10)
Label(prediccion_frame, text="Resultados de Detección:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
prediction_text_widget = Text(prediccion_frame, height=8, state=tk.DISABLED, wrap=tk.WORD, relief=tk.SUNKEN, bd=1, font=("Consolas", 9)) 
scrollbar_pred = Scrollbar(prediccion_frame, command=prediction_text_widget.yview)
prediction_text_widget.config(yscrollcommand=scrollbar_pred.set)
scrollbar_pred.pack(side=tk.RIGHT, fill=tk.Y)
prediction_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

status_bar_var = StringVar()
status_bar_var.set("Listo. El modelo por defecto se está cargando...")
status_bar = Label(window, textvariable=status_bar_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=5)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

do_load_yolo_model_threaded(DEFAULT_MODEL_NAME)

window.mainloop()