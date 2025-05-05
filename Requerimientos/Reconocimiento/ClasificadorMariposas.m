%% -------- CONFIGURACIÓN INICIAL --------
clear; clc; close all;

% --- Parámetros de Preprocesamiento ---
imageSize = [128 128];

% --- Umbrales HSV (Ajustados según tu especificación) ---
% Monarch (tonos naranja)
hue_monarch_range     = [0.03, 0.11];   % ~5–20/179 en OpenCV
sat_min_monarch       = 150/255;        % ≥150/255 ≃0.588
val_min_monarch       = 100/255;        % ≥100/255 ≃0.392

% Isabella (tonos amarillo)
hue_isabella_range    = [0.11, 0.20];   % ~20–35/179 en OpenCV
sat_min_isabella      = 100/255;        % ≥100/255 ≃0.392
val_min_isabella      = 100/255;        % ≥100/255 ≃0.392

% Zonas muy oscuras (negro / contornos)
val_max_black         =  50/255;        % ≤50/255  ≃0.196
% --- FIN Umbrales HSV Ajustados ---

% --- Parámetros de Depuración Visual ---
numDebugImages = 5; % Poner > 0 para ver imágenes y sus máscaras/features

% --- Parámetros de la Red Neuronal ---
hiddenLayerSizes = [25 10]; % Mantenemos el ajuste para 7 features
trainFcn = 'trainscg';
epochs = 2500;
goal = 0.005;
performFcn = 'crossentropy';

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

%% -------- 2. PREPROCESAMIENTO Y EXTRACCIÓN DE CARACTERÍSTICAS (7 ESCALARES) --------
tic;
disp('Iniciando preprocesamiento y extracción de 7 características ESCALARES...');

numDebugImages = min(numDebugImages, numImages);
if numDebugImages > 0
    indicesToShow = sort(randperm(numImages, numDebugImages));
    disp(['Se mostrarán mapas y pausarán para ', num2str(numDebugImages), ' imágenes aleatorias:']);
    disp(indicesToShow);
else
    indicesToShow = [];
end

numScalarFeatures = 7; % Usando 7 features
allScalarFeatures = zeros(numScalarFeatures, numImages);
allTargets = zeros(numClasses, numImages);

% --- BUCLE PRINCIPAL SOBRE IMÁGENES ORIGINALES ---
for i = 1:numImages
    % Leer Imagen Original
    img = readimage(imds, i);
    currentLabel = imds.Labels(i);

    % Redimensionar
    imgResized = imresize(img, imageSize, 'bicubic');

    % Preprocesamiento Básico (Color y Escala de Grises)
    imgGray = []; % Inicializar por si acaso
    if size(imgResized, 3) == 3 % Color
        imgHSV = rgb2hsv(imgResized);
        H=imgHSV(:,:,1); S=imgHSV(:,:,2); V=imgHSV(:,:,3);
        imgGray = rgb2gray(imgResized);
        if isfloat(imgGray), imgGrayUint8 = im2uint8(imgGray); else, imgGrayUint8 = imgGray; end
        if ~isfloat(imgGray), imgGray=im2double(imgGray); end
    else % Grayscale
        imgGray = imgResized;
        if isfloat(imgGray), imgGrayUint8 = im2uint8(imgGray); else, imgGrayUint8 = imgGray; end
        if ~isfloat(imgGray), imgGray=im2double(imgGray); end
        H=zeros(imageSize); S=zeros(imageSize); V=imgGray;
    end

    % --- Calcular Mapas Intermedios ---
    maskMonarchOrange = (H >= hue_monarch_range(1) & H <= hue_monarch_range(2) & S >= sat_min_monarch & V >= val_min_monarch);
    maskIsabellaYellow = (H >= hue_isabella_range(1) & H <= hue_isabella_range(2) & S >= sat_min_isabella & V >= val_min_isabella);
    edgeImg = edge(imgGray, 'Sobel');
    maskBlack = (V <= val_max_black); % Usa el nuevo umbral de negro
    try
        entropyImg = entropyfilt(imgGray);
    catch ME_entropy
        warning('Error en entropyfilt para imagen %d: %s. Usando 0.', i, ME_entropy.message);
        entropyImg = zeros(imageSize);
    end

    % --- Calcular Features Escalares (Originales + Textura) ---
    totalPixels = prod(imageSize);
    featOrangeRatio = sum(maskMonarchOrange(:))/totalPixels;
    featYellowRatio = sum(maskIsabellaYellow(:))/totalPixels;
    featEdgeDensity = sum(edgeImg(:))/totalPixels;
    featBlackRatio  = sum(maskBlack(:))/totalPixels;
    featMeanEntropy = mean(entropyImg(:));

    try
        glcm = graycomatrix(imgGrayUint8, 'Offset', [0 1], 'Symmetric', true);
        if isempty(glcm), stats.Contrast = 0; stats.Homogeneity = 0; else, stats = graycoprops(glcm, {'Contrast', 'Homogeneity'}); end
    catch ME_glcm
        warning('Error calculando GLCM/props para imagen %d: %s. Usando 0.', i, ME_glcm.message);
        stats.Contrast = 0; stats.Homogeneity = 0;
    end
    featContrast = stats.Contrast;
    featHomogeneity = stats.Homogeneity;

    % Almacenar las 7 características
    allScalarFeatures(:, i) = [featOrangeRatio; featYellowRatio; featEdgeDensity; featBlackRatio; featMeanEntropy; featContrast; featHomogeneity];

    % Crear Vector Objetivo
    targetVector = zeros(numClasses, 1);
    targetVector(strcmp(char(currentLabel), classNames)) = 1;
    allTargets(:, i) = targetVector;

    % --- BLOQUE DE DEPURACIÓN VISUAL ---
    if ismember(i, indicesToShow)
         figure('Name', ['Preproc Debug - Imagen Original ', num2str(i)], 'NumberTitle', 'off');
         subplot(2, 3, 1); imshow(imgResized); title(['Original/Resized ', num2str(i)]);
         subplot(2, 3, 2); imshow(maskMonarchOrange); title('Mask Nar.');
         subplot(2, 3, 3); imshow(maskIsabellaYellow); title('Mask Ama.');
         subplot(2, 3, 4); imshow(edgeImg); title('Bordes Sobel');
         subplot(2, 3, 5); imshow(maskBlack); title(['Mask Negra (V<=',num2str(val_max_black,'%.3f'),')']); % Muestra nuevo umbral
         subplot(2, 3, 6); imagesc(entropyImg); axis image off; colormap jet; colorbar; title('Entropía');
         drawnow;
         disp('-----------------------------------------------------');
         disp(['Mostrando MAPAS para imagen ORIGINAL: ', num2str(i)]);
         disp(['Etiqueta Real: ', char(currentLabel)]);
         fprintf('Features: O_R=%.3f, Y_R=%.3f, Edge=%.3f, Blk_R=%.3f, Entr=%.2f, Contr=%.2f, Homog=%.3f\n', ...
             featOrangeRatio, featYellowRatio, featEdgeDensity, featBlackRatio, featMeanEntropy, featContrast, featHomogeneity);
         disp('>>> Presiona tecla para continuar...');
         disp('-----------------------------------------------------');
         pause;
    end

    if mod(i, 50) == 0 || i == numImages
        fprintf('Procesamiento: Extraídas features para %d / %d imágenes...\n', i, numImages);
    end
end % Fin del bucle FOR
preprocessTime = toc;
disp(['Extracción de 7 features completada en ', num2str(preprocessTime, '%.2f'), ' segundos.']);


%% -------- 3. NORMALIZACIÓN DE CARACTERÍSTICAS (7 ESCALARES) --------
mu_scalar = mean(allScalarFeatures, 2);
sig_scalar = std(allScalarFeatures, 0, 2);
sig_scalar(sig_scalar < 1e-6) = 1e-6;
featuresScalarNormalized = (allScalarFeatures - mu_scalar) ./ sig_scalar;
disp('7 Características ESCALARES normalizadas.');

%% -------- 4. DEFINICIÓN Y ENTRENAMIENTO --------
disp('Configurando y entrenando red neuronal (Entrada = 7 Features)...');
net = patternnet(hiddenLayerSizes, trainFcn);
net.performFcn = performFcn;
try
    net = configure(net, featuresScalarNormalized, allTargets);
    disp('Red configurada con tamaño de entrada/salida.');
catch ME_conf
    error('Error al configurar la red: %s', ME_conf.message);
end
net.trainParam.epochs = epochs;
net.trainParam.goal = goal;
net.trainParam.showWindow = true;
net.trainParam.showCommandLine = true;
net.trainParam.time = inf;
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
disp(['>>> Iniciando Entrenamiento con 7 Features por imagen... <<<']);
[netTrained, tr] = train(net, featuresScalarNormalized, allTargets);
disp('Entrenamiento completado.');
disp(['Rendimiento final (',net.performFcn ,' en Validación): ', num2str(tr.best_vperf)]);
disp(['En Época: ', num2str(tr.best_epoch)]);
figure; plotperform(tr);

%% -------- 5. PRUEBA INTERACTIVA CON VARIAS IMÁGENES --------
disp(' ');
disp('--- INICIO PRUEBA INTERACTIVA (CON 7 FEATURES) ---');
numTestImagesToAsk = 5;

for test_iter = 1:numTestImagesToAsk
    fprintf('\n--- Prueba de Imagen Individual #%d de %d ---\n', test_iter, numTestImagesToAsk);
    prompt_title = sprintf('Selecciona imagen de prueba #%d/%d', test_iter, numTestImagesToAsk);
    [fileName, filePath] = uigetfile({'*.jpg;*.png;*.bmp;*.tif', 'Selecciona imagen'}, prompt_title);

    if isequal(fileName, 0) || isequal(filePath, 0)
        disp('Selección cancelada. Finalizando prueba interactiva.');
        break;
    end

    fullImagePath = fullfile(filePath, fileName);
    disp(['Probando imagen: ', fullImagePath]);
    try
        % --- Recalcular las 7 features escalares para prueba ---
        testImg = imread(fullImagePath);
        testImgResized = imresize(testImg, imageSize, 'bicubic');
        if size(testImgResized, 3) == 3
            testImgHSV = rgb2hsv(testImgResized);
            tH=testImgHSV(:,:,1); tS=testImgHSV(:,:,2); tV=testImgHSV(:,:,3);
            testImgGray = rgb2gray(testImgResized);
             if isfloat(testImgGray), testImgGrayUint8 = im2uint8(testImgGray); else, testImgGrayUint8 = testImgGray; end
             if ~isfloat(testImgGray), testImgGray=im2double(testImgGray); end
        else
            testImgGray = testImgResized;
             if isfloat(testImgGray), testImgGrayUint8 = im2uint8(testImgGray); else, testImgGrayUint8 = testImgGray; end
             if ~isfloat(testImgGray), testImgGray=im2double(testImgGray); end
            tH=zeros(imageSize); tS=zeros(imageSize); tV=testImgGray;
        end

        tMaskMonarch = (tH >= hue_monarch_range(1) & tH <= hue_monarch_range(2) & tS >= sat_min_monarch & tV >= val_min_monarch);
        tMaskIsabella = (tH >= hue_isabella_range(1) & tH <= hue_isabella_range(2) & tS >= sat_min_isabella & tV >= val_min_isabella);
        tEdgeImg = edge(testImgGray, 'Sobel');
        tMaskBlack = (tV <= val_max_black); % Usa el nuevo umbral
        try
            tEntropyImg = entropyfilt(testImgGray);
        catch ME_entropy_test
             warning('Error en entropyfilt para img prueba: %s. Usando 0.', ME_entropy_test.message);
             tEntropyImg = zeros(imageSize);
        end
         % Calcular GLCM features para prueba
        try
            tglcm = graycomatrix(testImgGrayUint8, 'Offset', [0 1], 'Symmetric', true);
            if isempty(tglcm), tstats.Contrast = 0; tstats.Homogeneity = 0; else, tstats = graycoprops(tglcm, {'Contrast', 'Homogeneity'}); end
        catch ME_glcm_test
            warning('Error calculando GLCM/props para img prueba: %s. Usando 0.', ME_glcm_test.message);
            tstats.Contrast = 0; tstats.Homogeneity = 0;
        end
        tFeatContrast = tstats.Contrast;
        tFeatHomogeneity = tstats.Homogeneity;

        tTotalPixels = prod(imageSize);
        tFeatOrangeRatio = sum(tMaskMonarch(:))/tTotalPixels;
        tFeatYellowRatio = sum(tMaskIsabella(:))/tTotalPixels;
        tFeatEdgeDensity = sum(tEdgeImg(:))/tTotalPixels;
        tFeatBlackRatio  = sum(tMaskBlack(:))/tTotalPixels;
        tFeatMeanEntropy = mean(tEntropyImg(:));

        % Crear vector de 7 features para prueba
        testFeatureVectorScalar = [tFeatOrangeRatio; tFeatYellowRatio; tFeatEdgeDensity; tFeatBlackRatio; tFeatMeanEntropy; tFeatContrast; tFeatHomogeneity];

        % Normalizar usando mu_scalar y sig_scalar (ahora tienen 7 filas)
        testFeaturesScalarNormalized = (testFeatureVectorScalar - mu_scalar) ./ sig_scalar;

        % --- Clasificar ---
        predictedScores = netTrained(testFeaturesScalarNormalized);
        [maxScore, predictedIndex] = max(predictedScores);
        predictedClassName = classNames{predictedIndex};

        % --- Mostrar Resultados ---
        figure; imshow(testImg);
        title({sprintf('Prueba #%d: %s', test_iter, fileName), ...
            ['Predicción: ', strrep(predictedClassName, '_', ' ')], ...
            ['Confianza: ', num2str(maxScore*100, '%.1f'), '%']}, 'Interpreter', 'none');

        disp(['Predicción para "', fileName, '": ', predictedClassName]);
        disp(['Scores (Probabilidades): ', num2str(predictedScores', '%.4f ')]);
        disp(['Input Features (Normalized): ', num2str(testFeaturesScalarNormalized', '%.3f ')]); % Muestra las 7

    catch ME
        disp('------------------- ERROR DURANTE LA PRUEBA -------------------');
        fprintf('Error al procesar/clasificar la imagen "%s":\n%s\n', fileName, ME.message);
        fprintf('En: %s (%s) Línea: %d\n', ME.stack(1).file, ME.stack(1).name, ME.stack(1).line);
        disp('-----------------------------------------------------------------');
    end
end % Fin del bucle FOR de pruebas

disp('--- FIN PRUEBA INTERACTIVA ---');
disp('Script finalizado.');