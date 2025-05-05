%% -------- CONFIGURACIÓN INICIAL --------
clear; clc; close all;

% --- Parámetros de Preprocesamiento ---
imageSize = [128 128];
% --- Umbrales HSV (Los que dieron buenos resultados) ---
hue_monarch_range     = [0.03, 0.11];
sat_min_monarch       = 150/255;
val_min_monarch       = 100/255;
hue_isabella_range    = [0.11, 0.20];
sat_min_isabella      = 100/255;
val_min_isabella      = 100/255;
val_max_black         =  50/255;

% --- Parámetros de Depuración Visual ---
numDebugImages = 0;

% --- Parámetros de la Red Neuronal MANUAL ---
numInputFeatures = 7;
numOutputClasses = 2;
hiddenLayerSizes_manual = [25 10];
arquitectura_bp = [numInputFeatures, hiddenLayerSizes_manual, numOutputClasses];

% --- Funciones de Activación y Derivadas (COMO STRINGS) ---
numWeightLayers = length(arquitectura_bp) - 1;
funciones_str = cell(1, numWeightLayers);
derivadas_str = cell(1, numWeightLayers);
for k = 1:numWeightLayers
    funciones_str{k} = '1./(1+exp(-net))'; % Sigmoide Logística
    derivadas_str{k} = 'act.*(1-act)';      % Derivada de Sigmoide
end
disp('Funciones de activación (Sigmoide) y derivadas definidas.');

% --- Parámetros de Entrenamiento MANUAL ---
alpha = 0.05;
beta = 0.8;
precision_bp = 0.05; % Detenerse si MSE/2 baja de esto (o maxEpocas)
maxEpocas_bp = 3000; % Aumentado límite por si tarda más en converger

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
% (Se mantiene igual que antes)
tic;
disp('Iniciando preprocesamiento y extracción de 7 características ESCALARES...');
% ... (código de cálculo de features idéntico al anterior, omitido por brevedad)...
numDebugImages = min(numDebugImages, numImages); if numDebugImages > 0, indicesToShow = sort(randperm(numImages, numDebugImages)); else, indicesToShow = []; end
numScalarFeatures = 7; allScalarFeatures = zeros(numScalarFeatures, numImages); allTargets = zeros(numClasses, numImages);
for i = 1:numImages
    img = readimage(imds, i); currentLabel = imds.Labels(i); imgResized = imresize(img, imageSize, 'bicubic'); imgGray = [];
    if size(imgResized, 3) == 3
        imgHSV = rgb2hsv(imgResized); H=imgHSV(:,:,1); S=imgHSV(:,:,2); V=imgHSV(:,:,3); imgGray = rgb2gray(imgResized);
        if isfloat(imgGray), imgGrayUint8 = im2uint8(imgGray); else, imgGrayUint8 = imgGray; end; if ~isfloat(imgGray), imgGray=im2double(imgGray); end
    else
        imgGray = imgResized; if isfloat(imgGray), imgGrayUint8 = im2uint8(imgGray); else, imgGrayUint8 = imgGray; end
        if ~isfloat(imgGray), imgGray=im2double(imgGray); end; H=zeros(imageSize); S=zeros(imageSize); V=imgGray;
    end
    maskMonarchOrange = (H >= hue_monarch_range(1) & H <= hue_monarch_range(2) & S >= sat_min_monarch & V >= val_min_monarch); maskIsabellaYellow = (H >= hue_isabella_range(1) & H <= hue_isabella_range(2) & S >= sat_min_isabella & V >= val_min_isabella);
    edgeImg = edge(imgGray, 'Sobel'); maskBlack = (V <= val_max_black);
    try entropyImg = entropyfilt(imgGray); catch, warning('Err Entr %d',i); entropyImg = zeros(imageSize); end
    totalPixels = prod(imageSize); featOrangeRatio = sum(maskMonarchOrange(:))/totalPixels; featYellowRatio = sum(maskIsabellaYellow(:))/totalPixels; featEdgeDensity = sum(edgeImg(:))/totalPixels; featBlackRatio = sum(maskBlack(:))/totalPixels; featMeanEntropy = mean(entropyImg(:));
    try glcm = graycomatrix(imgGrayUint8, 'Offset', [0 1], 'Symmetric', true); if isempty(glcm), stats.Contrast=0; stats.Homogeneity=0; else, stats = graycoprops(glcm, {'Contrast', 'Homogeneity'}); end; catch, warning('Err GLCM %d',i); stats.Contrast=0; stats.Homogeneity=0; end
    featContrast = stats.Contrast; featHomogeneity = stats.Homogeneity;
    allScalarFeatures(:, i) = [featOrangeRatio; featYellowRatio; featEdgeDensity; featBlackRatio; featMeanEntropy; featContrast; featHomogeneity];
    targetVector = zeros(numClasses, 1); targetVector(strcmp(char(currentLabel), classNames)) = 1; allTargets(:, i) = targetVector;
    if ismember(i, indicesToShow) % Debug Visual
         figure('Name', ['Preproc Debug - Imagen Original ', num2str(i)], 'NumberTitle', 'off'); subplot(2, 3, 1); imshow(imgResized); title(['Original/Resized ', num2str(i)]); subplot(2, 3, 2); imshow(maskMonarchOrange); title('Mask Nar.'); subplot(2, 3, 3); imshow(maskIsabellaYellow); title('Mask Ama.'); subplot(2, 3, 4); imshow(edgeImg); title('Bordes Sobel'); subplot(2, 3, 5); imshow(maskBlack); title(['Mask Negra (V<=',num2str(val_max_black,'%.3f'),')']); subplot(2, 3, 6); imagesc(entropyImg); axis image off; colormap jet; colorbar; title('Entropía'); drawnow;
         fprintf('Features: O_R=%.3f, Y_R=%.3f, Edge=%.3f, Blk_R=%.3f, Entr=%.2f, Contr=%.2f, Homog=%.3f\n', allScalarFeatures(:,i)); pause;
    end
    if mod(i, 50) == 0 || i == numImages, fprintf('.'); end
end
fprintf('\n'); preprocessTime = toc;
disp(['Extracción de 7 features completada en ', num2str(preprocessTime, '%.2f'), ' segundos.']);

%% -------- 3. NORMALIZACIÓN DE CARACTERÍSTICAS (7 ESCALARES) --------
mu_scalar = mean(allScalarFeatures, 2);
sig_scalar = std(allScalarFeatures, 0, 2);
sig_scalar(sig_scalar < 1e-6) = 1e-6;
featuresScalarNormalized = (allScalarFeatures - mu_scalar) ./ sig_scalar;
disp('7 Características ESCALARES normalizadas.');

%% -------- 4. DEFINICIÓN Y ENTRENAMIENTO (CON BP_neuronal_network) --------
disp('--- Usando implementación manual BP_neuronal_network ---');
disp('Creando instancia de la red BP...');
try
    miRed = BP_neuronal_network(arquitectura_bp, funciones_str, derivadas_str);
catch ME_create
    error('Error al crear el objeto BP_neuronal_network: %s', ME_create.message);
end

% Configurar ejes para gráfica de error (Opcional, tu clase lo usa)
hFigError = figure('Name', 'Error de Entrenamiento (Manual BP)');
hAxError = axes(hFigError);
title(hAxError, 'Progreso del Error'); % Título inicial

disp(['Iniciando entrenamiento manual BP (Alpha=', num2str(alpha), ...
      ', Beta=', num2str(beta), ', Precision=', num2str(precision_bp), ', MaxEpocas=', num2str(maxEpocas_bp),')...']);
try
    % Entrenar la red (usando tu clase)
    [redEntrenada, epocasFinales, errorFinal, historialError] = miRed.entrenar( ...
        featuresScalarNormalized, allTargets, alpha, precision_bp, beta, hAxError);

    disp('Entrenamiento manual BP completado.');
    fprintf('Épocas realizadas: %d\n', epocasFinales);
    fprintf('Error final (MSE/2): %.8g\n', errorFinal);

catch ME_train
    error('Error durante la ejecución de miRed.entrenar(): %s', ME_train.message);
end

% -------- 4.1 EVALUACIÓN POST-ENTRENAMIENTO --------
fprintf('\n--- Evaluación Post-Entrenamiento (sobre todos los datos) ---\n');
try
    % Obtener predicciones de la red entrenada sobre TODOS los datos de entrada
    disp('Calculando predicciones sobre el conjunto de datos completo...');
    outputsCompleto = redEntrenada.feedforward(featuresScalarNormalized);

    % Convertir salidas de la red y targets one-hot a índices de clase
    % La clase predicha es el índice de la neurona de salida con mayor activación
    [~, predictedIndices] = max(outputsCompleto, [], 1); % Índice del máximo en cada columna (predicción)
    [~, trueIndices] = max(allTargets, [], 1);          % Índice del '1' en cada columna (verdadera)

    % Calcular número de aciertos
    numCorrect = sum(predictedIndices == trueIndices);
    totalSamples = size(featuresScalarNormalized, 2); % numImages

    % Calcular y mostrar precisión
    trainingAccuracy = numCorrect / totalSamples;
    fprintf('Número de aciertos: %d de %d\n', numCorrect, totalSamples);
    fprintf('Precisión sobre el conjunto de datos completo: %.2f%%\n', trainingAccuracy * 100);

    % Opcional: Mostrar Matriz de Confusión Manualmente
    if exist('confusionchart', 'file') % Verifica si tienes la función (Stats Toolbox)
        figure;
        confusionchart(trueIndices, predictedIndices, classNames);
        title('Matriz de Confusión (Manual BP - Todo el Dataset)');
        disp('Mostrando matriz de confusión en una nueva figura...');
    else
        disp('Función confusionchart no encontrada (requiere Statistics and Machine Learning Toolbox).');
        % Podrías calcular la matriz manualmente si lo necesitas:
        % ConfMatManual = zeros(numClasses, numClasses);
        % for idx_sample = 1:totalSamples
        %     ConfMatManual(trueIndices(idx_sample), predictedIndices(idx_sample)) = ...
        %         ConfMatManual(trueIndices(idx_sample), predictedIndices(idx_sample)) + 1;
        % end
        % disp('Matriz de Confusión Manual (Filas=Real, Col=Predicción):');
        % disp(ConfMatManual);
    end

catch ME_eval
    warning('Error durante la evaluación post-entrenamiento: %s', ME_eval.message);
end
% -----------------------------------------------------

%% -------- 5. PRUEBA INTERACTIVA CON VARIAS IMÁGENES --------
% (Esta sección se mantiene igual que antes, usa redEntrenada)
disp(' ');
disp('--- INICIO PRUEBA INTERACTIVA (CON RED MANUAL BP) ---');
numTestImagesToAsk = 5;
% ... (resto del código de prueba interactiva idéntico al anterior) ...
for test_iter = 1:numTestImagesToAsk
    fprintf('\n--- Prueba de Imagen Individual #%d de %d ---\n', test_iter, numTestImagesToAsk);
    prompt_title = sprintf('Selecciona imagen de prueba #%d/%d', test_iter, numTestImagesToAsk);
    [fileName, filePath] = uigetfile({'*.jpg;*.png;*.bmp;*.tif', 'Selecciona imagen'}, prompt_title);
    if isequal(fileName, 0) || isequal(filePath, 0); disp('Selección cancelada.'); break; end
    fullImagePath = fullfile(filePath, fileName);
    disp(['Probando imagen: ', fullImagePath]);
    try
        testImg = imread(fullImagePath); testImgResized = imresize(testImg, imageSize, 'bicubic');
        if size(testImgResized, 3) == 3
            testImgHSV=rgb2hsv(testImgResized); tH=testImgHSV(:,:,1); tS=testImgHSV(:,:,2); tV=testImgHSV(:,:,3); testImgGray=rgb2gray(testImgResized);
            if isfloat(testImgGray), testImgGrayUint8=im2uint8(testImgGray); else, testImgGrayUint8=testImgGray; end; if ~isfloat(testImgGray), testImgGray=im2double(testImgGray); end
        else
            testImgGray=testImgResized; if isfloat(testImgGray), testImgGrayUint8=im2uint8(testImgGray); else, testImgGrayUint8=testImgGray; end
            if ~isfloat(testImgGray), testImgGray=im2double(testImgGray); end; tH=zeros(imageSize); tS=zeros(imageSize); tV=testImgGray;
        end
        tMaskMonarch=(tH >= hue_monarch_range(1) & tH <= hue_monarch_range(2) & tS >= sat_min_monarch & tV >= val_min_monarch);
        tMaskIsabella=(tH >= hue_isabella_range(1) & tH <= hue_isabella_range(2) & tS >= sat_min_isabella & tV >= val_min_isabella);
        tEdgeImg=edge(testImgGray, 'Sobel'); tMaskBlack=(tV <= val_max_black);
        try tEntropyImg=entropyfilt(testImgGray); catch, warning('Err Entr Test'); tEntropyImg=zeros(imageSize); end
        try tglcm=graycomatrix(testImgGrayUint8,'Offset',[0 1],'Symmetric',true); if isempty(tglcm),tstats.Contrast=0;tstats.Homogeneity=0;else,tstats=graycoprops(tglcm,{'Contrast','Homogeneity'});end; catch,warning('Err GLCM Test');tstats.Contrast=0;tstats.Homogeneity=0;end
        tFeatContrast=tstats.Contrast; tFeatHomogeneity=tstats.Homogeneity; tTotalPixels=prod(imageSize);
        tFeatOrangeRatio=sum(tMaskMonarch(:))/tTotalPixels; tFeatYellowRatio=sum(tMaskIsabella(:))/tTotalPixels; tFeatEdgeDensity=sum(tEdgeImg(:))/tTotalPixels; tFeatBlackRatio=sum(tMaskBlack(:))/tTotalPixels; tFeatMeanEntropy=mean(tEntropyImg(:));
        testFeatureVectorScalar=[tFeatOrangeRatio; tFeatYellowRatio; tFeatEdgeDensity; tFeatBlackRatio; tFeatMeanEntropy; tFeatContrast; tFeatHomogeneity];
        testFeaturesScalarNormalized = (testFeatureVectorScalar - mu_scalar) ./ sig_scalar;
        predictedScores = redEntrenada.feedforward(testFeaturesScalarNormalized);
        [maxScore, predictedIndex] = max(predictedScores); predictedClassName = classNames{predictedIndex};
        figure; imshow(testImg); title({sprintf('Prueba #%d: %s', test_iter, fileName), ['Predicción (BP Manual): ', strrep(predictedClassName, '_', ' ')], ['(Max Activación: ', num2str(maxScore, '%.3f'), ')']}, 'Interpreter', 'none');
        disp(['Predicción (BP Manual) para "', fileName, '": ', predictedClassName]); disp(['Salidas de la Red: ', num2str(predictedScores', '%.4f ')]); disp(['Input Features (Normalized): ', num2str(testFeaturesScalarNormalized', '%.3f ')]);
    catch ME
        disp('--- ERROR PRUEBA ---'); fprintf('Error: %s\nEn: %s (%s) Línea: %d\n', ME.message, ME.stack(1).file, ME.stack(1).name, ME.stack(1).line); disp('---');
    end
end
disp('--- FIN PRUEBA INTERACTIVA ---');
disp('Script finalizado.');