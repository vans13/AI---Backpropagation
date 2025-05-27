import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Label, Button, Frame, Canvas, PhotoImage, PanedWindow, Text, Scrollbar, StringVar, OptionMenu
from PIL import Image, ImageTk
import os
import cv2
from ultralytics import YOLO
import threading

class YoloObjectDetectorApp:
    """
    Clase principal para la aplicación de detección de objetos con YOLOv8.
    Gestiona la interfaz gráfica, la carga de modelos, la selección de archivos
    y el procesamiento de imágenes y videos.
    """
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Detector de Objetos con YOLOv8 - UniCundinamarca")
        master.geometry("1200x850")

        # --- Atributos de Configuración e Inicialización ---
        self.DEFAULT_MODEL_NAME = 'yolov8n.pt'
        self.MAX_CANVAS_WIDTH = 620
        self.MAX_CANVAS_HEIGHT = 480

        self.model = None
        self.current_loaded_model_name = ""
        self.logo_tk_ref = None # Referencia para la imagen del logo, evita recolección de basura.

        self.current_media_path = None
        self.current_media_type = None # "imagen" o "video"
        self.original_image_cv_rgb = None # Imagen original (OpenCV RGB) para procesamiento.
        self.tk_image_on_canvas = None # Imagen (PhotoImage) actualmente en el canvas.
        self.detections_for_hover = [] # Almacena detecciones para la funcionalidad hover.

        self.video_capture = None # Objeto VideoCapture de OpenCV.
        self.is_streaming_video_active = False # Controla el bucle de reproducción de video.
        self.video_frame_count = 0
        self.video_delay_ms = 30 # Retardo entre frames, calculado a partir de FPS.
        self.acumulador_objetos_video_global = {} # Acumula detecciones en videos.
        
        self.file_path_display_var = StringVar(master, "Ningún archivo seleccionado")
        self.media_info_var = StringVar(master, "Tipo: N/A\nDimensiones: N/A\nTamaño: N/A")
        self.status_bar_var = StringVar(master, "Listo.")
        self.model_status_var = StringVar(master, "Activo: (Ninguno)")
        self.selected_model_option_var = StringVar(master, self.DEFAULT_MODEL_NAME)

        self._crear_widgets()
        self._configurar_estilos_ttk()

        self.do_load_yolo_model_threaded(self.DEFAULT_MODEL_NAME) # Carga el modelo por defecto al iniciar.

    def _configurar_estilos_ttk(self):
        """Aplica estilos personalizados a los widgets ttk para una apariencia moderna."""
        style = ttk.Style()
        available_themes = style.theme_names()
        
        # Selección de tema base según disponibilidad en el sistema operativo.
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'vista' in available_themes: 
            style.theme_use('vista')
        elif 'aqua' in available_themes: 
            style.theme_use('aqua')
        
        # Definición de estilos específicos para diferentes tipos de widgets.
        style.configure("TButton", padding=6, relief="flat", font=("Arial", 10))
        style.configure("Bold.TButton", font=("Arial", 10, "bold"))
        style.configure("Accent.TButton", font=("Arial", 12, "bold"), background="#AED6F1")
        style.configure("TLabel", padding=2, font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 18, "bold"))
        style.configure("SubHeader.TLabel", font=("Arial", 11, "bold"))
        style.configure("Status.TLabel", relief=tk.SUNKEN, padding=5, anchor=tk.W)
        style.configure("Info.TLabel", anchor=tk.W, justify=tk.LEFT)
        style.configure("Path.TLabel", relief=tk.GROOVE, padding=5, anchor=tk.W)
        style.configure("TMenubutton", font=("Arial", 10), padding=5)


    def _crear_widgets(self):
        """Construye y posiciona todos los elementos de la interfaz gráfica."""
        
        # --- Sección Superior: Título y Logo ---
        top_frame_container = ttk.Frame(self.master, padding=(10, 5))
        top_frame_container.pack(side=tk.TOP, fill=tk.X)

        title_integrantes_frame = ttk.Frame(top_frame_container)
        title_integrantes_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(10,0))

        title_label = ttk.Label(title_integrantes_frame, text="Laboratorio IA: Detección de Objetos con YOLOv8", style="Header.TLabel")
        title_label.pack(pady=(0,5), anchor='w')
        
        subtitle_frame = ttk.Frame(title_integrantes_frame)
        subtitle_frame.pack(fill=tk.X, anchor='w')
        integrantes_label = ttk.Label(subtitle_frame, text="Presentado por: Juan Fuentes, Laura Latore, Duván Matallana", font=("Arial", 10))
        integrantes_label.pack(side=tk.LEFT, anchor='w')
        uni_label = ttk.Label(subtitle_frame, text=" - Universidad de Cundinamarca", font=("Arial", 10, "italic"))
        uni_label.pack(side=tk.LEFT, anchor='w')

        # Carga y visualización del logo.
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__)) # Directorio del script actual.
            logo_path = os.path.join(script_dir, "uni.png") # Ruta al archivo del logo.
            logo_pil = Image.open(logo_path)
            logo_resized_pil = logo_pil.resize((80, 80), Image.Resampling.LANCZOS)
            self.logo_tk_ref = ImageTk.PhotoImage(logo_resized_pil) # Guardar referencia.
            
            logo_label_widget = ttk.Label(top_frame_container, image=self.logo_tk_ref)
            logo_label_widget.image = self.logo_tk_ref # Asignar referencia al widget.
            logo_label_widget.pack(side=tk.RIGHT, padx=(0,20), pady=(0,5))
        except FileNotFoundError:
            print(f"ERROR CRÍTICO: No se encontró el archivo del logo en la ruta: {logo_path}")
        except Exception as e:
            print(f"Advertencia: No se pudo cargar el logo ('uni.png'). Error: {e}")
            placeholder_logo = ttk.Label(top_frame_container, text="[Logo]", width=10)
            placeholder_logo.pack(side=tk.RIGHT, padx=20, pady=(0,5))

        # --- Paneles Principales (Visualización y Controles) ---
        main_paned_window = PanedWindow(self.master, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bd=2, sashwidth=10, background="#ECECEC")
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5,10))

        left_panel = ttk.Frame(main_paned_window, padding=5, relief=tk.RIDGE, borderwidth=2)
        main_paned_window.add(left_panel, stretch="always", width=700, minsize=450) 
        ttk.Label(left_panel, text="Visualización del Medio:", style="SubHeader.TLabel").pack(pady=(5,5), anchor="w", padx=5)
        self.canvas = Canvas(left_panel, bg="lightgray", width=self.MAX_CANVAS_WIDTH, height=self.MAX_CANVAS_HEIGHT, relief=tk.SUNKEN, borderwidth=1)
        self.canvas.pack(pady=5, padx=5, expand=True, fill=tk.BOTH) 

        right_panel = ttk.Frame(main_paned_window, padding=10, relief=tk.RIDGE, borderwidth=2)
        main_paned_window.add(right_panel, stretch="never", minsize=450, width=480)
        control_frame = ttk.Frame(right_panel)
        control_frame.pack(fill=tk.BOTH, expand=True)

        # --- Controles en Panel Derecho ---
        entrada_frame = ttk.LabelFrame(control_frame, text="Archivo de Entrada", padding=10)
        entrada_frame.pack(fill=tk.X, pady=(10,5))
        file_path_label = ttk.Label(entrada_frame, textvariable=self.file_path_display_var, wraplength=300, style="Path.TLabel")
        file_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, pady=(2,5))
        self.select_button = ttk.Button(entrada_frame, text="Seleccionar", command=self.seleccionar_archivo_cmd, style="TButton")
        self.select_button.pack(side=tk.LEFT, padx=(5,0), pady=(2,5))

        media_info_frame = ttk.LabelFrame(control_frame, text="Información del Archivo", padding=10)
        media_info_frame.pack(fill=tk.X, pady=5)
        info_label = ttk.Label(media_info_frame, textvariable=self.media_info_var, style="Info.TLabel", wraplength=350)
        info_label.pack(anchor=tk.W, pady=(0,5), fill=tk.X)

        modelo_frame = ttk.LabelFrame(control_frame, text="Configuración del Modelo YOLOv8", padding=10)
        modelo_frame.pack(fill=tk.X, pady=5)
        model_select_load_frame = ttk.Frame(modelo_frame)
        model_select_load_frame.pack(fill=tk.X)
        yolo_model_options = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        self.model_option_menu = ttk.OptionMenu(model_select_load_frame, self.selected_model_option_var, self.DEFAULT_MODEL_NAME, *yolo_model_options)
        self.model_option_menu.config(width=18)
        self.model_option_menu.pack(side=tk.LEFT, pady=(2,5), fill=tk.X, expand=True)
        self.load_model_button = ttk.Button(model_select_load_frame, text="Cargar Modelo", command=self.cargar_modelo_seleccionado_cmd, style="TButton")
        self.load_model_button.pack(side=tk.LEFT, padx=(5,0), pady=(2,5))
        model_status_label = ttk.Label(modelo_frame, textvariable=self.model_status_var, font=("Arial", 9, "italic"), wraplength=350, justify=tk.LEFT)
        model_status_label.pack(fill=tk.X, pady=(5,0), anchor='w')

        self.ejecutar_button = ttk.Button(control_frame, text="Ejecutar Detección", command=self.ejecutar_cmd, style="Accent.TButton", state=tk.DISABLED)
        self.ejecutar_button.pack(pady=15, padx=0, fill=tk.X, ipady=8)

        prediccion_frame = ttk.LabelFrame(control_frame, text="Resultados de Detección", padding=10)
        prediccion_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.prediction_text_widget = Text(prediccion_frame, height=8, state=tk.DISABLED, wrap=tk.WORD, relief=tk.SUNKEN, bd=1, font=("Consolas", 9), yscrollcommand=True, padx=5, pady=5)
        scrollbar_pred = ttk.Scrollbar(prediccion_frame, orient=tk.VERTICAL, command=self.prediction_text_widget.yview)
        self.prediction_text_widget.config(yscrollcommand=scrollbar_pred.set)
        scrollbar_pred.pack(side=tk.RIGHT, fill=tk.Y)
        self.prediction_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Barra de Estado ---
        status_bar = ttk.Label(self.master, textvariable=self.status_bar_var, style="Status.TLabel")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0), ipady=3)

    # --- Métodos de Carga y Gestión del Modelo ---
    def do_load_yolo_model_threaded(self, model_name_to_load: str):
        """Inicia la carga del modelo YOLO en un hilo secundario para no bloquear la GUI."""
        self.status_bar_var.set(f"Iniciando carga de modelo: {model_name_to_load}...")
        self.load_model_button.config(state=tk.DISABLED)
        self.ejecutar_button.config(state=tk.DISABLED)
        self.master.update_idletasks()

        thread = threading.Thread(target=self._do_load_yolo_model_blocking, args=(model_name_to_load,))
        thread.daemon = True # Permite que el programa principal cierre aunque el hilo siga activo.
        thread.start()

    def _do_load_yolo_model_blocking(self, model_name_to_load: str):
        """Carga el modelo YOLO. Se ejecuta en un hilo secundario."""
        print(f"Intentando cargar modelo: {model_name_to_load}")
        try:
            temp_model = YOLO(model_name_to_load)
            # Verificación de la carga exitosa del modelo.
            if hasattr(temp_model, 'predictor') and hasattr(temp_model, 'names') and len(temp_model.names) > 0:
                self.model = temp_model
                self.current_loaded_model_name = model_name_to_load
                # Actualizaciones de GUI se delegan al hilo principal.
                self.master.after(0, lambda: self.status_bar_var.set(f"Modelo ({self.current_loaded_model_name}) cargado exitosamente."))
                self.master.after(0, lambda: self.model_status_var.set(f"Activo: {self.current_loaded_model_name}"))
                print(f"Modelo ({self.current_loaded_model_name}) cargado.")
                self.master.after(0, lambda: self.ejecutar_button.config(state=tk.NORMAL))
            else:
                raise Exception("El modelo cargado no es válido o no tiene nombres de clases.")
        except Exception as e:
            self.master.after(0, lambda e=e: self.status_bar_var.set(f"Error al cargar modelo {model_name_to_load}: {e}"))
            self.master.after(0, lambda: self.model_status_var.set(f"Error al cargar: {model_name_to_load}. Intente otro."))
            print(f"Error crítico al cargar el modelo {model_name_to_load}: {e}")
            self.model = None
            self.master.after(0, lambda: self.ejecutar_button.config(state=tk.DISABLED))
        finally:
            # Reactivar el botón de carga independientemente del resultado.
            self.master.after(0, lambda: self.load_model_button.config(state=tk.NORMAL))

    def cargar_modelo_seleccionado_cmd(self):
        """Obtiene el modelo seleccionado y llama a la función de carga."""
        model_to_load = self.selected_model_option_var.get()
        self.do_load_yolo_model_threaded(model_to_load)

    # --- Métodos de Selección y Carga de Archivos Multimedia ---
    def seleccionar_archivo_cmd(self):
        """Abre un diálogo para seleccionar un archivo (imagen o video) y lo prepara para visualización/procesamiento."""
        self.detener_stream_video() # Detener cualquier video activo.

        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo de Imagen o Video",
            filetypes=(("Archivos multimedia", "*.jpg *.jpeg *.png *.bmp *.gif *.mp4 *.avi *.mov *.mkv"),
                       ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif"),
                       ("Videos", "*.mp4 *.avi *.mov *.mkv"))
        )
        if not file_path:
            self.status_bar_var.set("Selección de archivo cancelada.")
            return

        self.current_media_path = file_path
        self.file_path_display_var.set(os.path.basename(file_path)) # Mostrar solo nombre del archivo.

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

        # Limpieza de estado previo.
        media_info_text = ""
        self.detections_for_hover.clear()
        self.prediction_text_widget.config(state=tk.NORMAL)
        self.prediction_text_widget.delete(1.0, tk.END)
        self.prediction_text_widget.config(state=tk.DISABLED)
        self.canvas.delete("all")

        if ext in image_extensions:
            self.current_media_type = "imagen"
            try:
                img_cv2 = cv2.imread(self.current_media_path)
                if img_cv2 is None:
                    raise Exception(f"OpenCV no pudo leer la imagen: {self.current_media_path}")
                self.original_image_cv_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(self.original_image_cv_rgb)
                
                canvas_w = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else self.MAX_CANVAS_WIDTH
                canvas_h = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else self.MAX_CANVAS_HEIGHT
                resized_pil = self._redimensionar_imagen_para_canvas(img_pil, canvas_w, canvas_h)
                
                if resized_pil:
                    self.tk_image_on_canvas = ImageTk.PhotoImage(resized_pil)
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image_on_canvas)
                    self.canvas.image = self.tk_image_on_canvas # Mantener referencia.
                    self.canvas.bind("<Motion>", self.hover_sobre_imagen_cmd) # Activar hover.
                else:
                    self.canvas.create_text(canvas_w/2, canvas_h/2, text="Error al redimensionar imagen", anchor="center", font=("Arial", 10))

                h_orig, w_orig, c_orig = self.original_image_cv_rgb.shape
                size_kb = os.path.getsize(self.current_media_path) / 1024
                media_info_text = f"Tipo: Imagen\nDimensiones: {w_orig}x{h_orig}x{c_orig}\nTamaño: {size_kb:.2f} KB"
                self.status_bar_var.set("Imagen cargada. Presione 'Ejecutar' para detectar.")
            except Exception as e:
                print(f"Error al cargar imagen: {e}")
                messagebox.showerror("Error de Imagen", f"No se pudo cargar o procesar la imagen:\n{e}")
                media_info_text = "Error al cargar imagen."
                self.status_bar_var.set("Error al cargar imagen.")
                self.current_media_type = None; self.original_image_cv_rgb = None
        elif ext in video_extensions:
            self.current_media_type = "video"
            self.original_image_cv_rgb = None 
            self.tk_image_on_canvas = None    
            self.canvas.delete("all") 
            canvas_w = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else self.MAX_CANVAS_WIDTH
            canvas_h = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else self.MAX_CANVAS_HEIGHT
            self.canvas.create_text(canvas_w/2, canvas_h/2, text="Video seleccionado.\nPresione 'Ejecutar'.", anchor="center", justify="center", font=("Arial",12))
            self.canvas.unbind("<Motion>") # Desactivar hover si estaba activo.
            try:
                temp_cap = cv2.VideoCapture(self.current_media_path)
                if not temp_cap.isOpened():
                    raise Exception("OpenCV no pudo abrir el video.")
                w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = temp_cap.get(cv2.CAP_PROP_FPS); temp_cap.release()
                size_kb = os.path.getsize(self.current_media_path) / 1024
                media_info_text = f"Tipo: Video\nDimensiones: {w}x{h}\nFPS: {fps:.2f}\nTamaño: {size_kb:.2f} KB"
                self.status_bar_var.set("Video cargado. Presione 'Ejecutar' para procesar.")
            except Exception as e:
                messagebox.showerror("Error de Video", f"No se pudo obtener información del video:\n{e}")
                media_info_text = "Error al obtener info del video."; self.current_media_type = None
        else:
            self.current_media_type = "desconocido"
            media_info_text = "Tipo de archivo no soportado."
            messagebox.showwarning("Archivo no Soportado", f"El archivo con extensión '{ext}' no es soportado.")
            self.status_bar_var.set("Tipo de archivo no soportado.")
        self.media_info_var.set(media_info_text)

    def _redimensionar_imagen_para_canvas(self, img_pil: Image.Image, target_width: int, target_height: int) -> Image.Image | None:
        """Redimensiona una imagen PIL manteniendo la relación de aspecto para ajustarse al canvas."""
        w, h = img_pil.size
        if w == 0 or h == 0 or target_width <= 0 or target_height <= 0 : return None
        escala = min(target_width / w, target_height / h)
        nuevo_ancho = int(w * escala)
        nueva_altura = int(h * escala)
        return img_pil.resize((nuevo_ancho, nueva_altura), Image.Resampling.LANCZOS)

    # --- Métodos de Procesamiento y Ejecución ---
    def ejecutar_cmd(self):
        """Inicia el procesamiento del medio (imagen o video) actualmente cargado."""
        self.detener_stream_video() # Precondición: detener cualquier video.
        self.detections_for_hover.clear() # Limpiar detecciones de hover previas.
        self.prediction_text_widget.config(state=tk.NORMAL)
        self.prediction_text_widget.delete(1.0, tk.END)
        self.prediction_text_widget.config(state=tk.DISABLED)
        
        # Restaurar visualización base antes del procesamiento.
        if self.current_media_type == "imagen" and self.tk_image_on_canvas and self.canvas:
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image_on_canvas)
            self.canvas.image = self.tk_image_on_canvas
        elif self.current_media_type == "video" and self.canvas:
            self.canvas.delete("all") # Limpiar para el video.

        # Validaciones previas al procesamiento.
        if not self.current_media_path:
            messagebox.showwarning("Archivo no seleccionado", "Por favor, seleccione un archivo multimedia.")
            return
        if self.model is None:
            messagebox.showerror("Modelo no cargado", "Por favor, cargue un modelo IA antes de ejecutar.")
            return

        # Derivación del procesamiento según el tipo de medio.
        if self.current_media_type == "imagen":
            self._procesar_imagen_seleccionada()
        elif self.current_media_type == "video":
            self._iniciar_procesamiento_video(self.current_media_path)
        else:
            messagebox.showerror("Error de Archivo", "Tipo de archivo no soportado o no se ha seleccionado un archivo válido.")
            self.status_bar_var.set("Error: Tipo de archivo no procesable.")
            
    def _procesar_imagen_seleccionada(self):
        """Realiza la detección de objetos en la imagen cargada."""
        if self.original_image_cv_rgb is None:
            self.status_bar_var.set("Error: No hay imagen original para procesar.")
            return
        # El modelo ya se verifica en ejecutar_cmd.

        self.status_bar_var.set(f"Procesando imagen con {self.current_loaded_model_name}...")
        self.master.update_idletasks() # Forzar actualización de la GUI.
        
        results = self.model(self.original_image_cv_rgb, verbose=False)
        if not results:
            self.status_bar_var.set("El modelo no devolvió resultados para la imagen.")
            return

        result = results[0] # Se asume un solo resultado para imágenes.
        self.detections_for_hover.clear()

        prediction_summary = f"Detecciones en Imagen ({self.current_loaded_model_name}):\n"
        if result.boxes and len(result.boxes) > 0:
            prediction_summary += f"  Total de objetos: {len(result.boxes)}\n"
            for i, box_data in enumerate(result.boxes):
                try: # Manejo de posibles errores de formato en los datos de detección.
                    x1, y1, x2, y2 = map(int, box_data.xyxy[0].cpu().numpy())
                    conf = box_data.conf[0].item()
                    cls_id = int(box_data.cls[0].item())
                    nombre_clase = self.model.names[cls_id]
                    self.detections_for_hover.append({'box': (x1, y1, x2, y2), 'name': nombre_clase, 'conf': conf})
                    prediction_summary += f"    {i+1}. {nombre_clase} (Conf: {conf:.2f})\n"
                except IndexError:
                    print(f"Advertencia: Error de formato en box_data (imagen): {box_data}")
                    continue 
        else:
            prediction_summary += "  No se detectaron objetos."
        
        self.prediction_text_widget.config(state=tk.NORMAL)
        self.prediction_text_widget.delete(1.0, tk.END)
        self.prediction_text_widget.insert(tk.END, prediction_summary)
        self.prediction_text_widget.config(state=tk.DISABLED)
        self.status_bar_var.set("Procesamiento de imagen completado. Mueva el cursor sobre la imagen.")

    def _iniciar_procesamiento_video(self, video_path: str):
        """Prepara e inicia el procesamiento de un archivo de video."""
        # El modelo ya se verifica en ejecutar_cmd.

        self.detener_stream_video() 
        self.acumulador_objetos_video_global.clear()
        self.video_frame_count = 0
        
        self.canvas.delete("all") 
        self.canvas.unbind("<Motion>")

        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error de Video", f"No se pudo abrir el video: {video_path}")
            canvas_w = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else self.MAX_CANVAS_WIDTH
            canvas_h = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else self.MAX_CANVAS_HEIGHT
            self.canvas.create_text(canvas_w/2, canvas_h/2, text="Error al cargar el video.", anchor="center", justify="center", font=("Arial",12))
            return

        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.video_delay_ms = int(1000 / fps) if fps > 0 else 33 # 33ms ~ 30 FPS por defecto.
        
        self.is_streaming_video_active = True
        self.status_bar_var.set(f"Reproduciendo video: {os.path.basename(video_path)} con {self.current_loaded_model_name}...")
        self._actualizar_frame_video() # Comienza el bucle de lectura de frames.

    def _actualizar_frame_video(self):
        """Lee, procesa y muestra un frame de video. Se llama recursivamente."""
        if not self.is_streaming_video_active or self.video_capture is None or self.model is None:
            return # Condiciones de parada del bucle.

        ret, frame = self.video_capture.read() # Leer un frame.
        if ret:
            self.video_frame_count += 1
            
            frame_rgb_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertir a RGB para YOLO.
            results = self.model(frame_rgb_original, verbose=False, stream=False)
            if not results: 
                self.master.after(self.video_delay_ms, self._actualizar_frame_video) # Continuar si no hay resultados.
                return 

            result = results[0]
            
            # Dibujar detecciones en el frame usando el método plot() de YOLO.
            plotted_frame_bgr = result.plot(img=frame.copy()) # plot() espera BGR y devuelve BGR.
            plotted_frame_rgb = cv2.cvtColor(plotted_frame_bgr, cv2.COLOR_BGR2RGB) # Convertir a RGB para Tkinter.
            img_pil = Image.fromarray(plotted_frame_rgb)
            
            canvas_w = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else self.MAX_CANVAS_WIDTH
            canvas_h = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else self.MAX_CANVAS_HEIGHT
            resized_pil = self._redimensionar_imagen_para_canvas(img_pil, canvas_w, canvas_h)

            if resized_pil:
                # Crear y mostrar la imagen del frame actual.
                current_tk_image = ImageTk.PhotoImage(resized_pil)
                self.canvas.delete("all") 
                self.canvas.create_image(0, 0, anchor=tk.NW, image=current_tk_image)
                self.canvas.image = current_tk_image # Referencia crucial para este frame.

                # Actualizar resumen de detecciones del frame.
                current_frame_summary = f"Frame: {self.video_frame_count} ({self.current_loaded_model_name})\n"
                if result.boxes and len(result.boxes) > 0:
                    num_objetos = len(result.boxes)
                    current_frame_summary += f"  Objetos detectados: {num_objetos}\n  Lista: ["
                    det_list = []
                    for box_data in result.boxes:
                        try: # Manejo de errores de formato.
                            nombre_clase = self.model.names[int(box_data.cls[0].item())]
                            confianza = box_data.conf[0].item()
                            det_list.append(f"{nombre_clase} ({confianza:.2f})")
                            
                            # Acumular detecciones globales del video.
                            if nombre_clase not in self.acumulador_objetos_video_global:
                                self.acumulador_objetos_video_global[nombre_clase] = {'total_confianza': 0.0, 'cantidad': 0}
                            self.acumulador_objetos_video_global[nombre_clase]['total_confianza'] += confianza
                            self.acumulador_objetos_video_global[nombre_clase]['cantidad'] += 1
                        except IndexError:
                            print(f"Advertencia: Error de formato en box_data (video): {box_data}")
                            continue
                    current_frame_summary += ", ".join(det_list) + "]" if det_list else " (formato inválido)"
                else:
                    current_frame_summary += "  No se detectaron objetos."

                self.prediction_text_widget.config(state=tk.NORMAL)
                self.prediction_text_widget.delete(1.0, tk.END)
                self.prediction_text_widget.insert(tk.END, current_frame_summary)
                self.prediction_text_widget.config(state=tk.DISABLED)

            # Programar la lectura del siguiente frame.
            self.master.after(self.video_delay_ms, self._actualizar_frame_video)
        else:
            # Fin del video o error de lectura.
            self.detener_stream_video()
            self.status_bar_var.set("Video finalizado o error al leer frame.")
            self._mostrar_resumen_final_video()
            
    def detener_stream_video(self):
        """Libera el objeto de captura de video y detiene el stream."""
        self.is_streaming_video_active = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        print("Video stream detenido.")

    def _mostrar_resumen_final_video(self):
        """Muestra un resumen de todas las detecciones acumuladas durante el video."""
        if self.prediction_text_widget is None: return

        video_name = os.path.basename(self.current_media_path) if self.current_media_path and self.current_media_type == "video" else "N/A"
        summary_text = f"Resumen Final del Video ({video_name} con {self.current_loaded_model_name}):\n"
        
        if not self.acumulador_objetos_video_global:
            summary_text += "  No se acumularon detecciones en este video."
        else:
            summary_text += "  Total de apariciones y Conf. Promedio:\n"
            # Ordenar objetos por cantidad de apariciones.
            objetos_ordenados = sorted(self.acumulador_objetos_video_global.items(), key=lambda item: item[1]['cantidad'], reverse=True)
            for nombre_clase, data in objetos_ordenados:
                cantidad = data['cantidad']
                conf_prom = data['total_confianza'] / cantidad if cantidad > 0 else 0
                summary_text += f"    - {nombre_clase:<20} | Visto: {cantidad:<5} | Conf. Prom.: {conf_prom:.2f}\n"
        
        self.prediction_text_widget.config(state=tk.NORMAL)
        self.prediction_text_widget.delete(1.0, tk.END)
        self.prediction_text_widget.insert(tk.END, summary_text)
        self.prediction_text_widget.config(state=tk.DISABLED)
        self.status_bar_var.set(f"Video ({video_name}) procesado. Mostrando resumen.")

    # --- Métodos de Interacción con la GUI (Hover) ---
    def hover_sobre_imagen_cmd(self, event: tk.Event):
        """Muestra información de detección al pasar el cursor sobre un objeto en la imagen."""
        if self.tk_image_on_canvas is None or self.original_image_cv_rgb is None or self.current_media_type != "imagen":
            return
        
        # Redibujar la imagen base para limpiar detecciones de hover anteriores.
        self.canvas.delete("hover_box") 
        self.canvas.delete("hover_text")
        # No es necesario redibujar toda la imagen base si solo se limpian los tags.
        # self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image_on_canvas) 

        if not self.detections_for_hover:
            return

        x_mouse, y_mouse = event.x, event.y
        try:
            orig_h, orig_w = self.original_image_cv_rgb.shape[:2]
            disp_w, disp_h = self.tk_image_on_canvas.width(), self.tk_image_on_canvas.height()
        except AttributeError: 
            return # Salir si los atributos de imagen no están disponibles.

        if disp_w == 0 or disp_h == 0 or orig_w == 0 or orig_h == 0: return # Evitar división por cero.

        # Calcular factores de escala entre la imagen original y la mostrada en el canvas.
        scale_w_orig_to_disp = disp_w / orig_w
        scale_h_orig_to_disp = disp_h / orig_h

        for det in self.detections_for_hover:
            x1_orig, y1_orig, x2_orig, y2_orig = det['box']
            conf = det['conf']; nombre_clase = det['name']
            
            # Escalar coordenadas de la detección a las dimensiones del canvas.
            x1_disp = int(x1_orig * scale_w_orig_to_disp); y1_disp = int(y1_orig * scale_h_orig_to_disp)
            x2_disp = int(x2_orig * scale_w_orig_to_disp); y2_disp = int(y2_orig * scale_h_orig_to_disp)

            # Verificar si el cursor está sobre la detección actual.
            if x1_disp <= x_mouse <= x2_disp and y1_disp <= y_mouse <= y2_disp:
                label_text = f"{nombre_clase} ({conf:.2f})"
                self.canvas.create_rectangle(x1_disp, y1_disp, x2_disp, y2_disp, outline="lime green", width=2, tags="hover_box")
                
                text_x = x1_disp
                text_y = y1_disp - 5 
                anchor_pos = tk.SW 
                self.canvas.create_text(text_x, text_y, anchor=anchor_pos, fill="lime green", font=("Arial", 10, "bold"), text=label_text, tags="hover_text")
                break # Mostrar solo la información de la primera detección encontrada.

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloObjectDetectorApp(root)
    root.mainloop()