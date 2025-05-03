% --- Limpieza inicial ---
clear; % Borra las variables del espacio de trabajo
clc;   % Limpia la ventana de comandos
close all; % Cierra todas las figuras abiertas

% --- 1. Pedir al usuario que seleccione la Imagen ---
% Define los tipos de archivo que se pueden seleccionar
filterSpec = {'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp;*.gif','Archivos de Imagen (*.jpg, *.jpeg, *.png, *.tif, *.bmp, *.gif)';
              '*.*', 'Todos los Archivos (*.*)'}; % Permite seleccionar cualquier tipo si es necesario
dialogTitle = 'Selecciona un archivo de imagen';

% Abre el cuadro de diálogo para obtener el nombre y la ruta del archivo
[fileName, pathName] = uigetfile(filterSpec, dialogTitle);

% Verificar si el usuario canceló la selección
if isequal(fileName, 0) || isequal(pathName, 0)
    disp('Operación cancelada por el usuario.');
    return; % Detiene la ejecución si el usuario cancela
end

% Construir la ruta completa al archivo seleccionado
nombreArchivoCompleto = fullfile(pathName, fileName); % fullfile es robusto para diferentes OS

% --- 2. Cargar la Imagen ---
try
    img = imread(nombreArchivoCompleto);
catch ME % Manejo de errores si no se puede leer la imagen (p.ej., archivo corrupto)
    fprintf('Error al cargar o leer la imagen: %s\n', ME.message);
    fprintf('Asegúrate que el archivo "%s" es una imagen válida y tienes permisos para leerla.\n', nombreArchivoCompleto);
    return; % Detiene la ejecución si hay un error
end

% --- 3. Mostrar la Imagen Original ---
figure; % Crea una nueva ventana para las figuras
subplot(2, 1, 1); % Área superior para la imagen
imshow(img);      % Muestra la imagen cargada
% Muestra el nombre del archivo en el título para más claridad
% 'Interpreter', 'none' evita problemas si el nombre tiene guiones bajos (_)
title(['Imagen Original: ', fileName], 'Interpreter', 'none');

% --- 4. Separar los Canales de Color (R, G, B) ---
% Una imagen a color en MATLAB es una matriz 3D: Alto x Ancho x 3 (Canales)
% Verificar si la imagen es a color (tiene 3 dimensiones)
if size(img, 3) == 3
    canalRojo = img(:,:,1); % Extrae el primer plano (Rojo)
    canalVerde = img(:,:,2); % Extrae el segundo plano (Verde)
    canalAzul = img(:,:,3); % Extrae el tercer plano (Azul)
else
    % Si la imagen no es a color (es escala de grises o indexada)
    fprintf('La imagen seleccionada no es una imagen a color RGB.\n');
    % Podrías optar por mostrar solo el histograma de la imagen en escala de grises
    % O simplemente detener la parte del histograma RGB. Por ahora, detenemos.
     title('Imagen Original (No es RGB)', 'Interpreter', 'none');
    return;
end

% --- 5. Calcular el Histograma para Cada Canal ---
% La función imhist calcula el histograma de una imagen en escala de grises (o un solo canal)
% Devuelve el conteo de píxeles para cada nivel de intensidad (bins)
[countsRojo, binsRojo] = imhist(canalRojo);
[countsVerde, binsVerde] = imhist(canalVerde);
[countsAzul, binsAzul] = imhist(canalAzul);

% --- 6. Mostrar los Histogramas RGB ---
subplot(2, 1, 2); % Selecciona la segunda área de la cuadrícula para graficar
% Grafica el histograma del canal rojo en color rojo
plot(binsRojo, countsRojo, 'Red', 'LineWidth', 1.5);
hold on; % Mantiene la gráfica actual para añadir las siguientes
% Grafica el histograma del canal verde en color verde
plot(binsVerde, countsVerde, 'Green', 'LineWidth', 1.5);
% Grafica el histograma del canal azul en color azul
plot(binsAzul, countsAzul, 'Blue', 'LineWidth', 1.5);
hold off; % Libera la gráfica (las siguientes llamadas a plot crearán una nueva)

% --- 7. Añadir Detalles a la Gráfica del Histograma ---
title('Histograma de Color RGB');       % Título del histograma
xlabel('Nivel de Intensidad (0-255)'); % Etiqueta del eje X
ylabel('Número de Píxeles');          % Etiqueta del eje Y
legend('Canal Rojo', 'Canal Verde', 'Canal Azul', 'Location', 'northeast'); % Leyenda y su posición
grid on; % Muestra una cuadrícula para facilitar la lectura
xlim([0 255]); % Asegura que el eje X vaya de 0 a 255 (rango típico de intensidad)

% --- Fin del script ---