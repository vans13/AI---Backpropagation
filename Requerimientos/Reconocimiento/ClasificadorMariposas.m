%% -------- CONFIGURACIÓN INICIAL --------
clear; clc; close all;

% --- Parámetros de Preprocesamiento ---
imageSize = [128 128];
hue_monarch_range = [0.07, 0.13]; % (Ajustar si es necesario)
sat_min_monarch = 0.55;          % (Ajustar si es necesario)
val_min_monarch = 0.4;           % (Ajustar si es necesario)
hue_isabella_range = [0.16, 0.22]; % (Ajustar si es necesario)
sat_min_isabella = 0.60;          % (Ajustar si es necesario)
val_min_isabella = 0.5;           % (Ajustar si es necesario)
val_max_black = 0.2;

% --- Parámetros de Depuración Visual ---
numDebugImages = 4; % Puedes poner 0 si ya no necesitas ver las máscaras

% --- Parámetros de la Red Neuronal ---
hiddenLayerSizes = [20]; % <<< Probando una capa oculta más grande
trainFcn = 'trainscg';
epochs = 250;        % <<< Más épocas (validación puede detener antes)
goal = 1e-6;         % <<< Objetivo más estricto para crossentropy

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

classNames = categories(imds.Labels);
numClasses = numel(classNames);
disp('Clases encontradas por imageDatastore:'); disp(classNames);
if numClasses ~= 2, warning('Diseñado para 2 clases, encontradas %d.', numClasses); end
if isempty(classNames), error('No se detectaron nombres de clase.'); end

%% -------- 2. PREPROCESAMIENTO Y EXTRACCIÓN DE CARACTERÍSTICAS (ESCALARES) --------
tic;
disp('Iniciando preprocesamiento y extracción de características ESCALARES...');
numDebugImages = min(numDebugImages, numImages);
if numDebugImages > 0
    indicesToShow = sort(randperm(numImages, numDebugImages));
    disp(['Se mostrarán mapas y pausarán ', num2str(numDebugImages), ' imágenes aleatorias:']);
    disp(indicesToShow);
else
    indicesToShow = []; % No mostrar ninguna si numDebugImages es 0
end

numScalarFeatures = 5;
allScalarFeatures = zeros(numScalarFeatures, numImages);
allTargets = zeros(numClasses, numImages);

for i = 1:numImages
    % --- Código de Preprocesamiento y Cálculo de Features ---
    % (Igual que antes, se asume correcto)
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
    maskMonarchOrange = (H >= hue_monarch_range(1) & H <= hue_monarch_range(2) & S >= sat_min_monarch & V >= val_min_monarch);
    maskIsabellaYellow = (H >= hue_isabella_range(1) & H <= hue_isabella_range(2) & S >= sat_min_isabella & V >= val_min_isabella);
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

    % --- Bloque de Depuración Visual (Opcional) ---
    if ismember(i, indicesToShow)
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
         disp(['Etiqueta Real: ', char(imds.Labels(i))]);
         disp(['Calculated Features: O_Ratio=', num2str(featOrangeRatio,'%.3f'), ...
               ', Y_Ratio=', num2str(featYellowRatio,'%.3f'), ', Edge=', num2str(featEdgeDensity,'%.3f'), ...
               ', Blk_Ratio=', num2str(featBlackRatio,'%.3f'), ', Entropy=', num2str(featMeanEntropy,'%.2f')]);
         disp('>>> Presiona tecla para continuar...');
         disp('-----------------------------------------------------');
         pause;
    end

    % --- Asignación de Target ---
    currentLabel = imds.Labels(i);
    targetVector = zeros(numClasses, 1);
    targetVector(strcmp(char(currentLabel), classNames)) = 1;
    allTargets(:, i) = targetVector;

    % Mostrar progreso general
    if mod(i, 50) == 0 || i == numImages
        fprintf('Preprocesamiento: Calculadas features para %d / %d imágenes...\n', i, numImages);
    end
end
preprocessTime = toc;
disp(['Preprocesamiento y cálculo de features completado en ', num2str(preprocessTime, '%.2f'), ' segundos.']);

% % --- Diagnósticos Opcionales (Comentados) ---
% disp('--- Verificación de Targets (POST-LOOP) ---');
% unique_targets = unique(allTargets', 'rows'); disp('Etiquetas únicas:'); disp(unique_targets);
% disp('Suma por columna:'); disp(unique(sum(allTargets, 1))); disp('Muestras por clase:'); disp(sum(allTargets, 2)');
% if size(unique_targets,1) ~= 2 || ~ismember([1 0], unique_targets, 'rows') || ~ismember([0 1], unique_targets, 'rows')
%      error('¡Problema con Targets!'); else; disp('Targets OK.'); end
% disp('--- Verificación de Features Escalares (Antes de Normalizar) ---');
% feature_std_dev = std(allScalarFeatures, 0, 2); disp('Std Dev Features:'); disp(feature_std_dev');
% if any(feature_std_dev < 1e-6); warning('Varianza casi cero en feature.'); end

%% -------- 3. NORMALIZACIÓN DE CARACTERÍSTICAS (ESCALARES) --------
mu_scalar = mean(allScalarFeatures, 2);
sig_scalar = std(allScalarFeatures, 0, 2);
sig_scalar(sig_scalar < 1e-6) = 1e-6;
featuresScalarNormalized = (allScalarFeatures - mu_scalar) ./ sig_scalar;
disp('Características ESCALARES normalizadas.');

%% -------- 4. DEFINICIÓN Y ENTRENAMIENTO --------
disp('Configurando red neuronal (Entrada = 5 Features Escalares)...');
net = patternnet(hiddenLayerSizes, trainFcn);

% --- VOLVER A CROSSENTROPY ---
disp('>>> Usando "crossentropy" como función de error (estándar para clasificación) <<<');
net.performFcn = 'crossentropy';

% --- CORRECCIÓN: Configurar la red ANTES de cualquier simulación (incluso diagnóstica) ---
% Esto asegura que la red conozca el tamaño de entrada/salida esperado.
try
    net = configure(net, featuresScalarNormalized, allTargets);
    disp('Red configurada con tamaño de entrada/salida.');
catch ME_conf
    error('Error al configurar la red con configure(): %s\nAsegúrate que featuresScalarNormalized y allTargets no estén vacíos.', ME_conf.message);
end
% ------------------------------------------------------------------------------------

% % --- Diagnóstico Opcional Salida Inicial (Comentado) ---
% disp('--- Salida de la Red ANTES de Entrenar ---');
% try
%     initial_outputs = net(featuresScalarNormalized);
%     disp('Primeras 5 columnas de la salida inicial:'); disp(initial_outputs(:, 1:min(5, size(initial_outputs, 2))));
%     if all(abs(initial_outputs(:)) < 1e-9); warning('¡Salida inicial cercana a CERO!'); end
% catch ME_sim
%     warning('No se pudo simular la red inicial: %s', ME_sim.message);
% end
% disp('------------------------------------------');

net.trainParam.epochs = epochs;
net.trainParam.goal = goal;
net.trainParam.showWindow = true;
net.trainParam.showCommandLine = true;
net.trainParam.time = inf;
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

disp('>>> Iniciando Entrenamiento (con CrossEntropy)... <<<');
[netTrained, tr] = train(net, featuresScalarNormalized, allTargets);
disp('Entrenamiento completado.');
disp(['Rendimiento final (CrossEntropy en Validación): ', num2str(tr.best_vperf)]);
disp(['En Época: ', num2str(tr.best_epoch)]);
figure; plotperform(tr); % Muestra la gráfica de rendimiento

%% -------- 5. PRUEBA CON UNA IMAGEN INDIVIDUAL --------
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
        % Recalcular features escalares para prueba...
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
        tMaskMonarch = (tH >= hue_monarch_range(1) & tH <= hue_monarch_range(2) & tS >= sat_min_monarch & tV >= val_min_monarch);
        tMaskIsabella = (tH >= hue_isabella_range(1) & tH <= hue_isabella_range(2) & tS >= sat_min_isabella & tV >= val_min_isabella);
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

        % Clasificar (La red ahora debería dar salidas tipo probabilidad)
        predictedScores = netTrained(testFeaturesScalarNormalized);
        [maxScore, predictedIndex] = max(predictedScores); % maxScore es la probabilidad estimada
        predictedClassName = classNames{predictedIndex};

        figure;
        imshow(testImg);
        title({['Imagen: ', fileName], ...
            ['Predicción: ', strrep(predictedClassName, '_', ' ')], ...
            ['Confianza: ', num2str(maxScore*100, '%.1f'), '%']}, ... % Ahora maxScore es más significativo
            'Interpreter', 'none');

        disp(['Predicción para "', fileName, '": ', predictedClassName]);
        disp(['Scores (Probabilidades): ', num2str(predictedScores', '%.4f ')]);
        disp(['Input Features (Normalized): ', num2str(testFeaturesScalarNormalized', '%.3f ')]);

    catch ME
        disp('------------------- ERROR -------------------');
        fprintf('Error al procesar/clasificar la imagen de prueba:\n%s\n', ME.message);
        fprintf('Archivo: %s\nLínea: %d\n', ME.stack(1).file, ME.stack(1).line);
        disp('-------------------------------------------');
    end
end
disp('Script finalizado.');