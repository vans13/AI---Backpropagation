%% main.m
% Script principal para entrenar y usar la red neural de clasificación de
% vocales + color sin usar toolbox de Deep Learning.

clear; clc; close all;


%% 1) Leer parámetros desde config.txt
configPath = 'config.txt';
config = BP_neuronal_network.leerConfig(configPath);
arquitectura = config.ultima_arquitectura;   % e.g. [784 N 10]
alpha        = config.alpha;                % coeficiente de aprendizaje
precision    = config.precision;            % error objetivo
if isfield(config,'beta')
    beta = config.beta;
else
    beta = 0;
end

%% 2) Cargar dataset
datasetPath = 'C:\Users\juan_\OneDrive - UNIVERSIDAD DE CUNDINAMARCA\Juanes\Trabajos 8vo semestre\Inteligencia artificial\DatasetVocales';
extensiones = {'*.png','*.jpg','*.jpeg'};
files = [];
for k=1:numel(extensiones)
    files = [files; dir(fullfile(datasetPath,extensiones{k}))]; %#ok<AGROW>
end
numPatrones = numel(files);
X = zeros(28*28, numPatrones);
Y = zeros(10,       numPatrones);

for i = 1:numPatrones
    % 2.1) Leer imagen
    img = imread(fullfile(files(i).folder, files(i).name));
    
    % 2.2) Determinar etiqueta (vocal + color) a partir del nombre de archivo
    %      Asumimos nombres tipo "A_azul.jpg", "e_verde.png", etc.
    [~, basename, ~] = fileparts(files(i).name);
    partes = split(basename, ' ');
    letra = lower(partes{1});
    color = lower(partes{2});
    
    % 2.3) Pre‑procesado: normalizar y reducir a 28×28 con un kernel de promedio
    img28 = resize_to_28(img);
    vec   = double(img28(:)) / 255;           % vector columna 784×1
    
    % 2.4) Construir vectores one‑hot de salida
    yvocal = zeros(5,1);
    switch letra
        case 'a', yvocal(1)=1;
        case 'e', yvocal(2)=1;
        case 'i', yvocal(3)=1;
        case 'o', yvocal(4)=1;
        case 'u', yvocal(5)=1;
    end
    ycolor = zeros(5,1);
    switch color
        case 'rojo',  ycolor(1)=1;
        case 'verde', ycolor(2)=1;
        case 'azul',  ycolor(3)=1;
        case {'blanco','blanca'}, ycolor(4)=1;
        case 'negro',  ycolor(5)=1;
        otherwise
         error('Color desconocido en %s: %s', files(i).name, color);
    end
    
    % 2.5) Guardar en matrices
    X(:,i) = vec;
    Y(:,i) = [yvocal; ycolor];   % ahora 5+5 = 10 filas
end

%% 3) Crear y entrenar la red
% Para todas las capas usamos sigmoide clásica y su derivada:
%% 3) Crear y entrenar la red
% Leemos desde disco los strings de activación y derivadas
funciones_sigmoide    = BP_neuronal_network.leerFunciones('funciones_activacion.txt');
derivadas_sigmoide     = BP_neuronal_network.leerFunciones('derivadas_activacion.txt');

% Comprobación rápida opcional:
assert( numel(funciones_sigmoide) == numel(derivadas_sigmoide), ...
        'Debe haber igual nº de activaciones y derivadas' );

% Instanciamos
net = BP_neuronal_network(arquitectura, funciones_sigmoide, derivadas_sigmoide);

% Entrenamos
[net, epoca, errFinal, histErr, histW, histB] = ...
    net.entrenar(X, Y, alpha, precision, beta);


fprintf('Entrenamiento finalizado en %d épocas. Error final=%.5f\n', epoca, errFinal);

%% 4) Graficar evolución del error
figure('Name','Error de entrenamiento');
plot(histErr,'LineWidth',1.5);
xlabel('Época');
ylabel('MSE/2');
title('Evolución del error');

%% 5) (Opcional) Graficar evolución de un peso concreto
% Por ejemplo, evolución del primer peso de W1
w1_hist = cellfun(@(Wcell) Wcell{1}(1), histW);
figure('Name','Peso W_{1}(1,1) a lo largo de las épocas guardadas');
plot(w1_hist,'-o');
xlabel('Checkpoint (cada 10 épocas)');
ylabel('Valor de W_{1}(1,1)');
title('Evolución del peso');

%% 6) Guardar pesos entrenados
net.guardarPesos('pesos_entrenados.txt');

%% 7) Función de predicción sobre nueva imagen
function [predLetra, predColor] = predecirImagen(rutaImg, net)
    img = imread(rutaImg);
    vec = double( reshape( resize_to_28(img), [], 1 ) ) / 255;
    y   = net.feedforward(vec);
    % separo vocales y colores
    [~, iv] = max(y(1:5)); 
    [~, ic] = max(y(6:10));
    letras = {'a','e','i','o','u'};
    colores = {'rojo','verde','azul','blanco','negro'};
    predLetra = letras{iv};
    predColor = colores{ic};
end

%% 8) Probar con nueva imagen  
% Abre un diálogo para que el usuario seleccione un fichero de imagen
[archivo, ruta] = uigetfile({'*.png;*.jpg;*.jpeg','Imágenes (*.png, *.jpg, *.jpeg)'}, ...
                            'Seleccione imagen para clasificar');
if isequal(archivo,0)
    disp('Selección cancelada por el usuario.');
else
    rutaCompleta = fullfile(ruta, archivo);

    % Llama a la función de predicción que ya declaraste
    [letraPred, colorPred] = predecirImagen(rutaCompleta, net);

    % Muestra por consola
    fprintf('Predicción:\n  → Vocal: %s\n  → Color: %s\n', letraPred, colorPred);

    % Y muestra la imagen con el título de su predicción
    figure('Name','Resultado de Clasificación');
    imshow(imread(rutaCompleta));
    title(sprintf('Letra: %s   |   Color: %s', letraPred, colorPred), 'FontSize', 14);
end


%% --- Función auxiliar: down‑sampling con kernel promedio ---
function out28 = resize_to_28(img)
    % si es RGB, convertimos a escala de grises
    if size(img,3)==3
        img = rgb2gray(img);
    end
    % aplicamos kernel de promedio y submuestreo
    [h, w] = size(img);
    kh = floor(h/28); kw = floor(w/28);
    kernel = ones(kh,kw) / (kh*kw);
    blurred = conv2(double(img), kernel, 'valid');
    % tomar cada kh-ésimo pixel
    out28 = blurred(1:kh:end, 1:kw:end);
    % en caso de no llegar justo a 28×28, recortamos o rellenamos
    out28 = imresize(out28, [28 28], 'bilinear');
end
