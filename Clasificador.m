% clasificarImagen.m
% Script “ligero” para cargar red ya entrenada y clasificar UNA imagen.

clear; clc; close all;

%% 1) Leer configuración y arquitectura
config = BP_neuronal_network.leerConfig('config.txt');
if ~isfield(config,'ultima_arquitectura')
    error('No se encontró el campo última_arquitectura en config.txt.');
end
arquitectura = config.ultima_arquitectura;

%% 2) Leer definiciones de funciones sigmoide y derivada
funciones    = BP_neuronal_network.leerFunciones('funciones_activacion.txt');
derivadas    = BP_neuronal_network.leerFunciones('derivadas_activacion.txt');
if numel(funciones) ~= numel(derivadas) || numel(funciones) ~= numel(arquitectura)-1
    error('Número de funciones no coincide con capas entrenables.');
end

%% 3) Instanciar la red y cargar pesos entrenados
net = BP_neuronal_network(arquitectura, funciones, derivadas);
net = net.cargarPesos('pesos_entrenados.txt');

%% 4) Pedir al usuario que elija una imagen
[archivo, carpeta] = uigetfile({'*.png;*.jpg;*.jpeg','Imágenes (*.png,*.jpg,*.jpeg)'}, ...
                               'Seleccione una imagen para clasificar');
if isequal(archivo,0)
    disp('Selección cancelada.'); 
    return;
end
rutaImg = fullfile(carpeta, archivo);

%% 5) Pre‑procesar la imagen
img = imread(rutaImg);
% Si es RGB → escala de grises
if size(img,3)==3
    img = rgb2gray(img);
end
% Kernel promedio + submuestreo
[h, w] = size(img);
kh = floor(h/28);  kw = floor(w/28);
kernel = ones(kh,kw)/(kh*kw);
blur = conv2(double(img), kernel, 'valid');
down = blur(1:kh:end, 1:kw:end);
img28 = imresize(down, [28 28], 'bilinear');
% Vector columna
vec = double(img28(:)) / 255;

%% 6) Clasificar con feed‑forward
salida = net.feedforward(vec);

% Separar vocal (1–5) y color (6–8)
[~, idxV] = max(salida(1:5));
[~, idxC] = max(salida(6:10));
vocales = {'a','e','i','o','u'};
colores = {'rojo','verde','azul','blanco', 'negro'};
predVocal = vocales{idxV};
predColor = colores{idxC};

%% 7) Mostrar resultados
fprintf('→ Predicción Vocal: %s\n', predVocal);
fprintf('→ Predicción Color: %s\n', predColor);

figure('Name','Clasificación de Imagen');
imshow(imread(rutaImg));
title(sprintf('Vocal: %s   |   Color: %s', predVocal, predColor), 'FontSize', 14);
