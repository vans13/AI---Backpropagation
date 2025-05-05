%% -------- CONFIGURACIÓN INICIAL --------
clear; clc; close all;

% --- Parámetros de Preprocesamiento ---
imageSize = [128 128];
hue_monarch_range = [0.07, 0.13]; % (Ajustar según pruebas)
sat_min_monarch = 0.55;          % (Ajustar según pruebas)
val_min_monarch = 0.4;           % (Ajustar según pruebas)
hue_isabella_range = [0.16, 0.22]; % (Ajustar según pruebas)
sat_min_isabella = 0.60;          % (Ajustar según pruebas)
val_min_isabella = 0.5;           % (Ajustar según pruebas)
val_max_black = 0.2;

% --- Parámetros de Depuración Visual ---
numDebugImages = 4;

% --- Parámetros de la Red Neuronal ---
hiddenLayerSizes = [10];
trainFcn = 'trainscg';
epochs = 100;
goal = 1e-5;

%% -------- 1. CARGA DE DATOS --------
disp('Selecciona la carpeta principal del Dataset...');
datasetBaseFolder = uigetdir('', 'Selecciona la carpeta del Dataset');
if isequal(datasetBaseFolder, 0), disp('Operación cancelada.'); return; end

imds = imageDatastore(datasetBaseFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
numImages = numel(imds.Files);
disp(['Número total de imágenes encontradas: ', num2str(numImages)]);
if numImages == 0, error('No se encontraron imágenes.'); end

classNames = categories(imds.Labels); % Cell array con los nombres de clase únicos
numClasses = numel(classNames);
disp('Clases encontradas por imageDatastore:'); disp(classNames); % Muestra las clases detectadas
if numClasses ~= 2, warning('Diseñado para 2 clases, encontradas %d.', numClasses); end
if isempty(classNames)
    error('No se detectaron nombres de clase (subcarpetas). Verifica la estructura del Dataset.');
end

%% -------- 2. PREPROCESAMIENTO Y EXTRACCIÓN DE CARACTERÍSTICAS (ESCALARES) --------
tic;
disp('Iniciando preprocesamiento y extracción de características ESCALARES...');
numDebugImages = min(numDebugImages, numImages);
indicesToShow = sort(randperm(numImages, numDebugImages));
disp(['Se mostrarán mapas y pausarán ', num2str(numDebugImages), ' imágenes aleatorias para depuración:']);
disp(indicesToShow);

numScalarFeatures = 5;
allScalarFeatures = zeros(numScalarFeatures, numImages);
allTargets = zeros(numClasses, numImages); % Inicializa a ceros

% --- BUCLE PRINCIPAL DE PREPROCESAMIENTO ---
for i = 1:numImages
    % Leer y Preprocesar Imagen (código igual que antes, omitido por brevedad)
    img = readimage(imds, i);
    imgResized = imresize(img, imageSize, 'bicubic');
    if size(imgResized, 3) == 3
        imgHSV = rgb2hsv(imgResized);
        H = imgHSV(:,:,1); S = imgHSV(:,:,2); V = imgHSV(:,:,3);
        imgGray = rgb2gray(imgResized);
    else
        H = zeros(imageSize); S = zeros(imageSize); V = double(imgResized)/255.0;
        imgGray = imgResized;
        warning('Imagen %d (%s) es escala de grises.', i, imds.Files{i});
    end
    maskMonarchOrange = (H >= hue_monarch_range(1) & H <= hue_monarch_range(2) & ...
        S >= sat_min_monarch & V >= val_min_monarch);
    maskIsabellaYellow = (H >= hue_isabella_range(1) & H <= hue_isabella_range(2) & ...
        S >= sat_min_isabella & V >= val_min_isabella);
    edgeImg = edge(imgGray, 'Sobel');
    maskBlack = (V <= val_max_black);
    entropyImg = entropyfilt(imgGray);
    totalPixels = numel(imgGray);
    featOrangeRatio = sum(maskMonarchOrange(:)) / totalPixels;
    featYellowRatio = sum(maskIsabellaYellow(:)) / totalPixels;
    featEdgeDensity = sum(edgeImg(:)) / totalPixels;
    featBlackRatio  = sum(maskBlack(:)) / totalPixels;
    featMeanEntropy = mean(entropyImg(:));
    allScalarFeatures(:, i) = [featOrangeRatio; featYellowRatio; featEdgeDensity; featBlackRatio; featMeanEntropy];

    % --- Crear Vector Objetivo (One-Hot Encoding) ---
    currentLabel = imds.Labels(i); % Tipo categorical
    targetVector = zeros(numClasses, 1); % Vector de ceros [0; 0]

    % --- DIAGNÓSTICO DETALLADO DE TARGETS (para las primeras 5 imágenes) ---
    if i <= 5
        disp(['--- Debug Target Iteración ', num2str(i), ' ---']);
        fprintf('Archivo: %s\n', imds.Files{i});
        fprintf('currentLabel (categorical): '); disp(currentLabel);
        fprintf('classNames (cellstr): '); disp(classNames'); % Muestra en columna
        % --- Comparación usando conversión explícita a char ---
        currentLabelStr = char(currentLabel); % Convertir a string
        comparison_result = strcmp(currentLabelStr, classNames);
        fprintf('Resultado de strcmp(''%s'', classNames): ', currentLabelStr); disp(comparison_result');
        if ~any(comparison_result)
             warning('¡NO SE ENCONTRÓ COINCIDENCIA PARA ESTA ETIQUETA!');
        end
        disp('------------------------------------');
    end
    % --- FIN DIAGNÓSTICO DETALLADO ---

    % --- Asignación de Target con Conversión Explícita ---
    targetVector(strcmp(char(currentLabel), classNames)) = 1; % Usa char() para asegurar comparación de strings
    allTargets(:, i) = targetVector; % Asigna el vector [1;0] o [0;1] (o [0;0] si falla strcmp)


    % --- BLOQUE DE DEPURACIÓN VISUAL (Muestra Mapas) ---
    if ismember(i, indicesToShow)
        % (código de visualización igual que antes, omitido por brevedad)
         figure('Name', ['Preproc Debug - Imagen ', num2str(i)], 'NumberTitle', 'off');
         subplot(2, 3, 1); imshow(imgResized); title(['Orig. ', num2str(i)]);
         subplot(2, 3, 2); imshow(maskMonarchOrange); title('Mask Nar.');
         subplot(2, 3, 3); imshow(maskIsabellaYellow); title('Mask Ama.');
         subplot(2, 3, 4); imshow(edgeImg); title('Bordes Sobel');
         subplot(2, 3, 5); imshow(maskBlack); title(['Mask Negra (V<=',num2str(val_max_black),')']);
         subplot(2, 3, 6); imagesc(entropyImg); axis image off; colormap jet; colorbar; title('Entropía');
         drawnow;
         disp('-----------------------------------------------------');
         disp(['Mostrando MAPAS para imagen ALEATORIA: ', num2str(i)]);
         % ... (resto de disp omitido por brevedad) ...
         disp('>>> Presiona tecla para continuar...');
         disp('-----------------------------------------------------');
         pause;
    end
    % --- FIN DEL BLOQUE DE DEPURACIÓN ---

    % Mostrar progreso general
    if mod(i, 50) == 0 || i == numImages
        fprintf('Preprocesamiento: Calculadas features para %d / %d imágenes...\n', i, numImages);
    end
end
preprocessTime = toc;
disp(['Preprocesamiento y cálculo de features completado en ', num2str(preprocessTime, '%.2f'), ' segundos.']);

% --- DIAGNÓSTICO 1: Verificar Targets (SE MANTIENE) ---
disp('--- Verificación de Targets (POST-LOOP) ---');
unique_targets = unique(allTargets', 'rows');
disp('Etiquetas únicas encontradas:');
disp(unique_targets);
disp('Suma por columna (debería ser todo 1s):');
disp(unique(sum(allTargets, 1)));
disp('Número de muestras por clase:');
disp(sum(allTargets, 2)');
disp('-----------------------------------------');
% Verifica si ahora sí tenemos las etiquetas correctas
if size(unique_targets,1) ~= 2 || ~ismember([1 0], unique_targets, 'rows') || ~ismember([0 1], unique_targets, 'rows')
     error('¡Problema AÚN detectado en la matriz de Targets! Revisa la salida de depuración del bucle y los nombres de carpeta.');
else
    disp('¡Verificación de Targets parece CORRECTA!');
end


% --- DIAGNÓSTICO 2: Verificar Varianza de Features (SE MANTIENE) ---
disp('--- Verificación de Features Escalares (Antes de Normalizar) ---');
feature_std_dev = std(allScalarFeatures, 0, 2);
disp('Desviación Estándar de cada feature (Naranja, Amarilla, Borde, Negra, Entropía):');
disp(feature_std_dev');
if any(feature_std_dev < 1e-6)
    warning('Varianza casi cero detectada en alguna característica escalar.');
end
disp('------------------------------------------------------------------');

%% -------- 3. NORMALIZACIÓN DE CARACTERÍSTICAS (ESCALARES) --------
mu_scalar = mean(allScalarFeatures, 2);
sig_scalar = std(allScalarFeatures, 0, 2);
sig_scalar(sig_scalar < 1e-6) = 1e-6;
featuresScalarNormalized = (allScalarFeatures - mu_scalar) ./ sig_scalar;
disp('Características ESCALARES normalizadas.');

%% -------- 4. DEFINICIÓN Y ENTRENAMIENTO (CON DIAGNÓSTICOS) --------
disp('Configurando red neuronal (Entrada = 5 Features Escalares)...');
net = patternnet(hiddenLayerSizes, trainFcn);

% --- MANTENER MSE POR AHORA HASTA CONFIRMAR QUE LAS EPOCAS AVANZAN ---
disp('>>> USANDO "mse" COMO FUNCIÓN DE ERROR PARA DIAGNÓSTICO <<<');
net.performFcn = 'mse';

net.trainParam.epochs = epochs;
net.trainParam.goal = goal;
net.trainParam.showWindow = true;
net.trainParam.showCommandLine = true;
net.trainParam.time = inf;
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% --- DIAGNÓSTICO 4: VER SALIDA INICIAL DE LA RED (SE MANTIENE) ---
disp('--- Salida de la Red ANTES de Entrenar (Usando MSE) ---');
try
    initial_outputs = net(featuresScalarNormalized); % Simular con datos normalizados
    disp('Primeras 5 columnas de la salida inicial:');
    disp(initial_outputs(:, 1:min(5, size(initial_outputs, 2))));
    if all(initial_outputs(:) == 0)
        warning('¡La salida inicial de la red es CERO!');
    elseif size(unique(initial_outputs', 'rows'), 1) == 1
         warning('¡La salida inicial de la red es CONSTANTE!');
    end
catch ME_sim
    warning('No se pudo simular la red inicial: %s', ME_sim.message);
end
disp('----------------------------------------------------');

disp('>>> Iniciando Entrenamiento (con MSE)... <<<');
[netTrained, tr] = train(net, featuresScalarNormalized, allTargets);
disp('Entrenamiento completado.');
disp(['Rendimiento final (MSE en Validación): ', num2str(tr.best_vperf)]);
disp(['En Época: ', num2str(tr.best_epoch)]); % <<< ¿ES MAYOR QUE 0 AHORA?
figure; plotperform(tr);

%% -------- 5. PRUEBA CON UNA IMAGEN INDIVIDUAL (CON 5 FEATURES) --------
% (Código igual que antes, omitido por brevedad)
disp(' ');
disp('Selecciona una imagen individual para probar la clasificación...');
[fileName, filePath] = uigetfile({'*.jpg;*.png;*.bmp;*.tif', 'Selecciona imagen'}, ...
    'Selecciona imagen para clasificar');
if isequal(fileName, 0)
    disp('No se seleccionó imagen para prueba.');
else
    fullImagePath = fullfile(filePath, fileName);
    disp(['Probando imagen: ', fullImagePath]);
    try
        % Recalcular features escalares para prueba... (código igual que antes)
        testImg = imread(fullImagePath);
        testImgResized = imresize(testImg, imageSize, 'bicubic');
        if size(testImgResized, 3) == 3
            testImgHSV = rgb2hsv(testImgResized);
            tH = testImgHSV(:,:,1); tS = testImgHSV(:,:,2); tV = testImgHSV(:,:,3);
            testImgGray = rgb2gray(testImgResized);
        else
            tH = zeros(imageSize); tS = zeros(imageSize); tV = double(testImgResized)/255.0;
            testImgGray = testImgResized;
        end
        tMaskMonarch = (tH >= hue_monarch_range(1) & tH <= hue_monarch_range(2) & ...
            tS >= sat_min_monarch & tV >= val_min_monarch);
        tMaskIsabella = (tH >= hue_isabella_range(1) & tH <= hue_isabella_range(2) & ...
            tS >= sat_min_isabella & tV >= val_min_isabella);
        tEdgeImg = edge(testImgGray, 'Sobel');
        tMaskBlack = (tV <= val_max_black);
        tEntropyImg = entropyfilt(testImgGray);
        tTotalPixels = numel(testImgGray);
        tFeatOrangeRatio = sum(tMaskMonarch(:)) / tTotalPixels;
        tFeatYellowRatio = sum(tMaskIsabella(:)) / tTotalPixels;
        tFeatEdgeDensity = sum(tEdgeImg(:)) / tTotalPixels;
        tFeatBlackRatio  = sum(tMaskBlack(:)) / tTotalPixels;
        tFeatMeanEntropy = mean(tEntropyImg(:));
        testFeatureVectorScalar = [tFeatOrangeRatio; tFeatYellowRatio; tFeatEdgeDensity; tFeatBlackRatio; tFeatMeanEntropy];
        testFeaturesScalarNormalized = (testFeatureVectorScalar - mu_scalar) ./ sig_scalar;

        % Clasificar
        predictedOutputs = netTrained(testFeaturesScalarNormalized);
        [~, predictedIndex] = max(predictedOutputs);
        predictedClassName = classNames{predictedIndex};
        confidence_approx = max(predictedOutputs);

        figure;
        imshow(testImg);
        title({['Imagen: ', fileName], ...
            ['Predicción: ', strrep(predictedClassName, '_', ' ')], ...
            ['(Score Max: ', num2str(confidence_approx, '%.2f'), ')']}, ...
            'Interpreter', 'none');

        disp(['Predicción para "', fileName, '": ', predictedClassName]);
        disp(['Scores (Salida Cruda de la Red): ', num2str(predictedOutputs', '%.4f ')]);
        disp(['Input Features (Normalized): ', num2str(testFeaturesScalarNormalized', '%.3f ')]);

    catch ME
        disp('------------------- ERROR -------------------');
        fprintf('Error al procesar/clasificar la imagen de prueba:\n%s\n', ME.message);
        fprintf('Archivo: %s\nLínea: %d\n', ME.stack(1).file, ME.stack(1).line);
        disp('-------------------------------------------');
    end
end
disp('Script finalizado.');