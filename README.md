[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020a+-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://github.com/ultralytics/yolov8)

> **Un kit de herramientas educativo para explorar los fundamentos de la IA** — desde el entrenamiento de redes neuronales en MATLAB hasta la detección de objetos en tiempo real con Python y YOLO.

---

## 📖 Tabla de Contenidos

- [Visión General](#-visión-general)
- [¿Qué Encontrarás Aquí?](#-qué-encontrarás-aquí)
- [¿Para Quién es Este Proyecto?](#-para-quién-es-este-proyecto)
- [Guía de Inicio Rápido](#-guía-de-inicio-rápido)
- [Guías de los Componentes](#-guías-de-los-componentes)
  - [1. Redes Neuronales en MATLAB](#1-redes-neuronales-en-matlab)
  - [2. Algoritmo de Backpropagation](#2-algoritmo-de-backpropagation)
  - [3. Detección de Objetos con YOLO](#3-detección-de-objetos-con-yolo)
- [Solución de Problemas](#-solución-de-problemas)
- [Cómo Contribuir](#-cómo-contribuir)
- [Recursos Adicionales](#-recursos-adicionales)
- [Licencia](#-licencia)

---

## 🎯 Visión General

¡Bienvenido a **AI - Backpropagation**! Este proyecto es más que solo código — es una puerta de entrada para entender la inteligencia artificial. 

**El Problema que Resolvemos:** Muchos principiantes enfrentan una brecha entre la teoría abstracta de la IA y su implementación práctica. Este repositorio cierra esa brecha con código claro que desmitifica conceptos complejos.

**Nuestra Misión:** Hacer la IA accesible, cercana y empoderadora para aprendices de todos los niveles.

---

## 🔍 ¿Qué Encontrarás Aquí?

Este repositorio contiene tres componentes educativos:

| Componente | Tecnología | Lo Que Aprenderás |
|-----------|-----------|-------------------|
| 🧠 **Redes Neuronales** | MATLAB | Comprensión visual de cómo se construyen y entrenan las redes neuronales |
| ⚙️ **Backpropagation** | MATLAB | El algoritmo fundamental que permite a las redes aprender de los datos |
| 👁️ **Detección de Objetos** | Python + YOLO | Visión por computadora en tiempo real para detectar objetos en imágenes y video |

---

## 👥 ¿Para Quién es Este Proyecto?

- **Estudiantes de Secundaria:** ¿Curioso por la IA? Comienza aquí con proyectos prácticos y visuales
- **Principiantes en ML:** Conecta conceptos teóricos con implementación real
- **Desarrolladores Curiosos:** Introducción rápida a redes neuronales en MATLAB e implementación de YOLO en Python

---

## 🚀 Guía de Inicio Rápido

### Requisitos Previos

Antes de comenzar, asegúrate de tener instalado lo siguiente:

| Software | Versión | Propósito | Enlace de Descarga |
|----------|---------|-----------|-------------------|
| **Git** | Última | Clonar este repositorio | [git-scm.com](https://git-scm.com/) |
| **MATLAB** | R2020a+ | Entrenamiento de redes neuronales | [mathworks.com](https://www.mathworks.com/) |
| **Python** | 3.8+ | Backpropagation y YOLO | [python.org](https://www.python.org/) |

### Instalación

#### Paso 1: Abre tu Terminal

La terminal (también llamada **Command Prompt** en Windows, **Terminal** en macOS/Linux) te permite dar instrucciones a tu computadora escribiendo comandos.

- **Windows:** Presiona `Win + R`, escribe `cmd`, presiona Enter
- **macOS:** Presiona `Cmd + Espacio`, escribe `terminal`, presiona Enter
- **Linux:** Presiona `Ctrl + Alt + T`

#### Paso 2: Clona el Repositorio

Copia y pega este comando en tu terminal:

```bash
git clone https://github.com/vans13/AI---Backpropagation.git
```

**¿Qué hace este comando?** `git clone` descarga todos los archivos del proyecto a tu computadora.

**Alternativa:** Si esto falla, descarga manualmente el archivo ZIP desde GitHub:
1. Ve a la página del repositorio
2. Haz clic en el botón verde `<> Code`
3. Selecciona `Download ZIP`
4. Extrae el archivo ZIP

#### Paso 3: Navega a la Carpeta del Proyecto

```bash
cd AI---Backpropagation
```

**¿Qué hace este comando?** Cambia tu directorio actual a la carpeta del proyecto.

✅ **¡Todo listo!** El código ahora está en tu computadora.

---

## 💻 Guías de los Componentes

### 1. Redes Neuronales en MATLAB

> **Concepto Clave:** MATLAB es como una calculadora súper poderosa que nos permite ver visualmente cómo las redes neuronales aprenden. Perfecto para principiantes porque abstrae las matemáticas complejas y se enfoca en la estructura del modelo y su rendimiento.

#### Configuración

1. **Abre MATLAB** en tu computadora

2. **Navega a la carpeta del proyecto** usando la barra de direcciones de MATLAB o la ventana de comandos:
   ```matlab
   cd AI---Backpropagation/matlab_models
   ```

3. **Verifica los Toolboxes Requeridos** — Escribe `ver` en la consola y busca:
   - ✓ Neural Network Toolbox
   - ✓ Deep Learning Toolbox
   
   **¿Falta algún toolbox?** Ve a la pestaña `HOME` → `Add-Ons` → `Get Add-Ons`, busca el toolbox e instálalo.

#### Ejecutando el Modelo

1. **Abre el script principal:** `App_Mariposas.mlapp`

2. **Ejecuta el código:**
   - Haz clic en el botón **Run** en el editor de MATLAB, O
   - Escribe en la consola:
     ```matlab
     ## Usa la aplicación .mlapp para poder entrenar y testear el modelo
     App_Mariposas.mlapp
     ```

3. **Observa los Resultados:** MATLAB generará ventanas de visualización:
   - **Gráfico de Rendimiento:** Muestra cómo el error disminuye con cada época de entrenamiento
   - **Matriz de Confusión:** Tabla visual que muestra la precisión de clasificación (diagonal fuerte = buen rendimiento)

#### Experimenta y Aprende

¡La mejor manera de aprender es experimentando! Intenta modificar estos parámetros en `.mlapp`:

```matlab
hiddenLayerSize = 10;  % Prueba 5, 20 o 50 — ¿más neuronas siempre es mejor?
learningRate = 0.01;   % Prueba 0.1 o 0.001 — ¿qué pasa con la velocidad de entrenamiento?
```

**💡 Consejo:** ¡Haz un cambio a la vez para entender su efecto!

---

### 2. Algoritmo de Backpropagation

> **Concepto Clave:** Imagina que estás aprendiendo a lanzar una pelota a una canasta. Cada vez que fallas, tu cerebro calcula qué hiciste mal (ángulo, fuerza) y ajusta tu próximo lanzamiento. Backpropagation es el algoritmo matemático que hace exactamente esto para las redes neuronales — calcula los errores de predicción y los propaga hacia atrás a través de la red para ajustar las conexiones y "aprender".

#### Ejecutando el Algoritmo

1. **Ejecuta el script principal: `App_Ent_Backpropagation.mlapp`**
   ```matlab
   App_Ent_Backpropagation.mlapp
   ```

2. **Observa el Proceso de Aprendizaje:**
   - El script genera un gráfico mostrando cómo la pérdida (error) disminuye durante las épocas de entrenamiento
   - La precisión final del modelo se imprime en la consola
   - **¡Ver la curva descender = observar el aprendizaje de la red!**

---

### 3. Detección de Objetos con YOLO

> **Concepto Clave:** YOLO (You Only Look Once) es un modelo de IA increíblemente rápido y eficiente que puede "ver" a través de una cámara o imagen y decirte qué objetos hay y dónde están — ¡todo en una fracción de segundo! A diferencia de otros métodos, YOLO mira la imagen completa una sola vez para hacer sus predicciones, de ahí su nombre y velocidad.

#### Configuración Detallada del Entorno

Esta es la configuración más compleja, pero seguir estos pasos cuidadosamente asegurará que todo funcione correctamente.

1. **Navega a la carpeta de YOLO:**
   ```bash
   cd RedYolov8
   ```

2. **Crea un Entorno Virtual** (práctica recomendada):
   
   Un entorno virtual es como una "caja de arena" — las librerías instaladas aquí no afectarán otros proyectos de Python.
   
   ```bash
   python -m venv venv
   ```

3. **Activa el Entorno Virtual:**

   Debes activar este entorno cada vez que trabajes en el proyecto. Verás `(venv)` aparecer al inicio de tu línea de terminal.

   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Instala las Dependencias:**
   
   Con el entorno activado, instala las librerías requeridas:
    ```bash
    pip install torch torchvision
    ```
    pip install ultralytics
    pip install opencv-python
    pip install pillow
   
     **¿Qué son estas?**
     - `torch`: El motor de aprendizaje profundo
     - `ultralytics`: Proporciona la implementación de YOLO que usaremos
     - `opencv-python`: Para procesamiento de video e imágenes
     - `pillow`: Para manipulación de imágenes en la interfaz gráfica
     
     **Nota:** `tkinter` viene incluido con Python por defecto, no necesitas instalarlo.

#### Ejecutando la Aplicación de Detección

Esta implementación incluye una **interfaz gráfica (GUI)** que hace la detección mucho más intuitiva:

##### Iniciar la Aplicación
  ```bash
  python PrediccionYolo.py
  ```
  Usar la Interfaz Gráfica
  
  Seleccionar Modelo: En la interfaz, elige el modelo YOLOv8 que deseas usar:
  
  yolov8n.pt - Nano (más rápido, menor precisión)
  yolov8s.pt - Small (equilibrado)
  yolov8m.pt - Medium
  yolov8l.pt - Large
  yolov8x.pt - Extra Large (más lento, mayor precisión)
  
  Cargar Modelo: Haz clic en "Cargar Modelo" y espera a que se cargue
  Seleccionar Archivo: Haz clic en "Seleccionar" para elegir una imagen o video
  Ejecutar Detección: Presiona "Ejecutar Detección"
  Ver Resultados:
    Para imágenes: Los objetos detectados se mostrarán en el panel izquierdo. Pasa el cursor sobre ellos para ver detalles
    Para videos: La detección se ejecutará en tiempo real frame por frame, mostrando un resumen al finalizar
  
  Nota sobre el Logo: La aplicación intenta cargar logo_unicundi.png. Si no lo tienes, simplemente aparecerá una advertencia en consola pero la aplicación funcionará normalmente.


## 🛠 Solución de Problemas

### Problemas Generales (Git)

<details>
<summary><b>Error:</b> <code>git clone</code> falla o da error de autenticación</summary>

**Causa:** Problema de red o configuración de Git.

**Solución:** Descarga manualmente:
1. Ve a la página del repositorio en GitHub
2. Haz clic en el botón verde `<> Code`
3. Selecciona `Download ZIP`
4. Extrae el archivo ZIP en tu computadora
</details>

### Problemas con MATLAB

<details>
<summary><b>Error:</b> <code>Undefined function or variable 'nombre_de_funcion'</code></summary>

**Causa:** MATLAB no puede encontrar el archivo porque no estás en la carpeta correcta.

**Solución:** 
1. Verifica tu directorio actual: `pwd` en la consola de MATLAB
2. Navega a la carpeta correcta: `cd ruta/a/matlab_models`
</details>

<details>
<summary><b>Error:</b> Mensaje diciendo que se requiere un "Toolbox"</summary>

**Causa:** Falta el Neural Network Toolbox o Deep Learning Toolbox.

**Solución:**
1. En MATLAB, ve a la pestaña `HOME` → `Add-Ons` → `Get Add-Ons`
2. Busca el toolbox requerido por nombre
3. Instálalo
</details>

### Problemas con Python/YOLO

<details>
<summary><b>Error:</b> <code>ModuleNotFoundError: No module named 'torch'</code> (o 'cv2', 'PIL')</summary>

**Causa:** Las librerías necesarias no están instaladas, o el entorno virtual no está activado.

**Solución:**
1. Asegúrate de que tu entorno virtual esté activado (deberías ver `(venv)` en la terminal)
2. Si no está activado, actívalo primero
3. Luego ejecuta:
```bash
   pip install torch torchvision ultralytics opencv-python pillow
```
</details>
<details>
<summary><b>Error:</b> <code>_tkinter.TclError</code> o problemas con la interfaz gráfica</summary>
Causa: Tkinter no está instalado o configurado correctamente.
Solución:

Windows/macOS: Tkinter viene con Python. Reinstala Python desde python.org
Linux (Ubuntu/Debian):
```bash
  sudo apt-get install python3-tk
```
Linux (Fedora):
```bash
  sudo dnf install python3-tkinter
```
</details>
<details>
<summary><b>Error:</b> <code>CUDA out of memory</code></summary>
Causa: El modelo es demasiado grande para la memoria de tu GPU.
Solución: En la interfaz, selecciona un modelo más pequeño como yolov8n.pt o yolov8s.pt antes de cargar.
</details>
<details>
<summary><b>Problema:</b> La detección con video es muy lenta</summary>
Causa: El procesamiento en tiempo real consume muchos recursos.
Soluciones:

Usa el modelo más pequeño: yolov8n.pt
Cierra otras aplicaciones que consuman recursos
Si tienes GPU NVIDIA, asegúrate de tener CUDA instalado para aceleración por hardware

</details>
<details>
<summary><b>Advertencia:</b> "Logo no cargado ('logo_unicundi.png')"</summary>
Causa: El archivo del logo no está en la carpeta del proyecto.
Solución: Esto es solo una advertencia. La aplicación funcionará perfectamente sin el logo. Si deseas añadirlo, coloca un archivo PNG llamado logo_unicundi.png en la misma carpeta que PrediccionYolo.py.
</details>
```

### **🤝 Cómo Contribuir**
¡Las contribuciones son el corazón del código abierto y son muy bienvenidas! ¿Encontraste un error? ¿Tienes una idea? ¿Quieres mejorar la documentación?
Cómo Contribuir

Haz Fork de este repositorio (haz clic en el botón "Fork" en la esquina superior derecha)
Crea una rama para tu característica:

```bash
  git checkout -b feature/CaracteristicaAsombrosa
```

Haz commit de tus cambios:

```bash
  git commit -m 'Agregar alguna CaracteristicaAsombrosa'
```

Sube tu rama:

```bash
  git push origin feature/CaracteristicaAsombrosa
```

Abre un Pull Request para revisión

** Ideas para Contribuir **

📊 Agregar más ejemplos o conjuntos de datos
📝 Mejorar la documentación o añadir explicaciones
🌍 Traducir el README a otros idiomas
⚡ Optimizar el código existente
🐛 Reportar bugs a través de Issues


### 📚 Recursos
¿Quieres aprender más? Aquí tienes recursos excelentes para continuar tu viaje en IA:
Redes Neuronales

3Blue1Brown - Neural Networks — Explicaciones visuales e intuitivas (¡altamente recomendado!)
Google's ML Crash Course — Curso gratuito y práctico

YOLO y Visión por Computadora

Ultralytics YOLOv8 Docs — Documentación oficial
PyImageSearch — Tutoriales de visión por computadora

MATLAB

MATLAB Onramp — Tutorial interactivo gratuito


### 📜 Licencia
Este proyecto está distribuido bajo la Licencia MIT.
En términos sencillos, eres libre de:

✅ Usar el código para cualquier propósito (incluyendo uso comercial)
✅ Modificar el código como desees
✅ Distribuir el código

Consulta el archivo LICENSE para más detalles.

### ** 🙏 Agradecimientos **
Este proyecto se construye sobre los hombros de gigantes:

Ultralytics — Por su increíble implementación de YOLOv8
MathWorks — Por proporcionar MATLAB para educación e investigación
Comunidad Open Source de ML — Por compartir conocimiento e impulsar la IA


### 📬 Contacto y Soporte

Autor: vans13
Repositorio: AI---Backpropagation

¿Necesitas Ayuda?
La mejor manera de obtener ayuda es abrir un Issue en este repositorio. ¡Esto permite que otros con el mismo problema puedan encontrar la solución!

<div align="center">
Hecho con ❤️ para la comunidad de aprendizaje de IA
⭐ ¡Dale estrella a este repo si te ayudó! ⭐
Última Actualización: Octubre 2025 | Versión 1.0.0
</div>
