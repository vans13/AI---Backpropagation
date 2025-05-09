% =========================================================================
% Script Principal: CNN Básica para Clasificación de Mariposas
% Implementación con componentes manuales y BP_neuronal_network para FC
% SIN USAR DEEP LEARNING TOOLBOX para las capas de red o entrenamiento.
% Autores: Juan Fuentes, Duvan matallana, Laura Latorre
% =========================================================================
clear; clc; close all;
rng('default'); % Para reproducibilidad

fprintf('--- Inicio del Script de Clasificación de Mariposas ---\n');

% --- 0. Configuración Inicial ---
dataset_root = 'dataset/'; % Cambia esto a la ruta de tu dataset
species = {'Danaus_plexippus', 'Euides_isabella'};
labels_map = containers.Map(species, {0, 1}); % 0 para Danaus, 1 para Euides
img_size = [128, 128]; % Tamaño fijo para redimensionar imágenes
train_ratio = 0.8;       % 80% para entrenamiento, 20% para prueba

% --- 1. Interacción con el Usuario para Configuración ---
fprintf('\n--- Configuración del Preprocesamiento y Red ---\n');

% Selección de Kernels Convolucionales
predefined_kernels = define_kernels_cnn(); % Helper function
kernel_names = fieldnames(predefined_kernels);
fprintf('Kernels Convolucionales Predefinidos Disponibles:\n');
for k_idx = 1:length(kernel_names)
    fprintf('  %d: %s\n', k_idx, kernel_names{k_idx});
end
selected_kernel_indices_str = input('Seleccione uno o más kernels por número (separados por comas, ej: 1,3): ', 's');
selected_kernel_indices = str2num(selected_kernel_indices_str); %#ok<ST2NM>
chosen_kernels_list = {};
if isempty(selected_kernel_indices) || any(selected_kernel_indices < 1 | selected_kernel_indices > length(kernel_names))
    fprintf('Selección de kernel inválida o vacía. Usando Kernel de Identidad por defecto.\n');
    chosen_kernels_list{1} = predefined_kernels.Identity;
else
    for k_idx = 1:length(selected_kernel_indices)
        chosen_kernels_list{end+1} = predefined_kernels.(kernel_names{selected_kernel_indices(k_idx)});
    end
end
fprintf('Kernels seleccionados: %s\n', strjoin(kernel_names(selected_kernel_indices(selected_kernel_indices <= length(kernel_names) & selected_kernel_indices >=1 )), ', '));


% Selección de Filtros de Color (Opcional)
color_filter_options = {'None', 'Grayscale', 'RedEmphasis', 'GreenEmphasis', 'BlueEmphasis', 'InvertColor'};
fprintf('\nFiltros de Color Disponibles:\n');
for cf_idx = 1:length(color_filter_options)
    fprintf('  %d: %s\n', cf_idx, color_filter_options{cf_idx});
end
selected_filter_idx_str = input(sprintf('Seleccione un filtro de color por número (1 para %s): ', color_filter_options{1}), 's');
selected_filter_idx = str2double(selected_filter_idx_str);
if isnan(selected_filter_idx) || selected_filter_idx < 1 || selected_filter_idx > length(color_filter_options)
    fprintf('Selección de filtro inválida. Usando "%s".\n', color_filter_options{1});
    selected_filter_idx = 1;
end
chosen_color_filter = color_filter_options{selected_filter_idx};
fprintf('Filtro de color seleccionado: %s\n', chosen_color_filter);

% Configuración de Capas Ocultas FC
fc_hidden_layers_str = input(['\nIngrese neuronas para las capas ocultas FC (separadas por comas, ej: 64,32 o 128): '], 's');
fc_hidden_neurons = str2num(fc_hidden_layers_str); %#ok<ST2NM>
if isempty(fc_hidden_neurons)
    fc_hidden_neurons = [64]; % Valor por defecto
    fprintf('Entrada inválida. Usando por defecto: [%s]\n', num2str(fc_hidden_neurons));
end
fprintf('Neuronas en capas ocultas FC: [%s]\n', num2str(fc_hidden_neurons));

% Parámetros de Entrenamiento
fprintf('\n--- Configuración del Entrenamiento ---\n');
alpha_str = input('Tasa de Aprendizaje (alpha, ej: 0.01): ', 's');
alpha = str2double(alpha_str);
if isnan(alpha) || alpha <= 0; alpha = 0.01; fprintf('Alpha inválido, usando %f\n', alpha); end

beta_str = input('Coeficiente de Momento (beta, ej: 0.9, 0 para ninguno): ', 's');
beta = str2double(beta_str);
if isnan(beta) || beta < 0 || beta >= 1; beta = 0; fprintf('Beta inválido, usando %f (sin momento)\n', beta); end
if beta > 0; fprintf('Usando Beta (Momento): %f\n', beta); else; fprintf('No se usará Beta (Momento).\n');end


max_epochs_str = input('Número Máximo de Épocas (ej: 1000): ', 's');
max_epochs = str2double(max_epochs_str);
if isnan(max_epochs) || max_epochs <= 0; max_epochs = 1000; fprintf('Épocas inválidas, usando %d\n', max_epochs); end

desired_precision_str = input('Precisión Deseada (error objetivo, ej: 0.001): ', 's');
desired_precision = str2double(desired_precision_str);
if isnan(desired_precision) || desired_precision <= 0; desired_precision = 0.001; fprintf('Precisión inválida, usando %f\n', desired_precision); end


% --- 2. Carga y Preparación Inicial de Datos ---
fprintf('\n--- Cargando y Preparando Datos ---\n');
[all_images_paths, all_labels_numeric] = load_and_label_images_cnn(dataset_root, species, labels_map);
if isempty(all_images_paths)
    error('No se cargaron imágenes. Verifique la ruta del dataset y la estructura de carpetas.');
end
num_total_images = length(all_images_paths);
fprintf('Total de imágenes cargadas: %d\n', num_total_images);

% Dividir en entrenamiento y prueba
cv = cvpartition(num_total_images, 'HoldOut', 1 - train_ratio);
idx_train = training(cv);
idx_test = test(cv);

train_image_paths = all_images_paths(idx_train);
train_labels = all_labels_numeric(idx_train);
test_image_paths = all_images_paths(idx_test);
test_labels = all_labels_numeric(idx_test);

fprintf('Imágenes de entrenamiento: %d, Imágenes de prueba: %d\n', length(train_image_paths), length(test_image_paths));

% --- 3. Preprocesamiento de Imágenes y Extracción de Características CNN ---
% Esto incluye: Filtro de color, Redimensionamiento, Convolución, ReLU, Pooling, Aplanado.
pool_params.size = [2, 2];      % Tamaño de la ventana de pooling
pool_params.stride = 2;         % Stride del pooling
activation_func = @relu_manual_cnn; % Función de activación ReLU manual

fprintf('\n--- Preprocesando Datos de Entrenamiento y Extrayendo Características ---\n');
X_train_mlp = preprocess_and_extract_features_cnn(train_image_paths, chosen_color_filter, img_size, chosen_kernels_list, activation_func, pool_params);

fprintf('\n--- Preprocesando Datos de Prueba y Extrayendo Características ---\n');
X_test_mlp = preprocess_and_extract_features_cnn(test_image_paths, chosen_color_filter, img_size, chosen_kernels_list, activation_func, pool_params);

if isempty(X_train_mlp) || isempty(X_test_mlp)
    error('La extracción de características no produjo datos. Revise el preprocesamiento.');
end
fprintf('Dimensiones de características de entrenamiento para MLP: [%d, %d]\n', size(X_train_mlp,1), size(X_train_mlp,2));
fprintf('Dimensiones de características de prueba para MLP: [%d, %d]\n', size(X_test_mlp,1), size(X_test_mlp,2));

% Asegurar formato de etiquetas para BP_neuronal_network: [num_outputs x num_samples]
% Para clasificación binaria con 1 neurona de salida (sigmoide), las etiquetas deben ser 0 o 1.
Y_train_mlp = train_labels; % Asumiendo que train_labels es [1 x num_train_samples]
Y_test_mlp = test_labels;   % Asumiendo que test_labels es [1 x num_test_samples]
if size(Y_train_mlp,1) ~= 1; Y_train_mlp = Y_train_mlp'; end % Asegurar que sea fila
if size(Y_test_mlp,1) ~= 1; Y_test_mlp = Y_test_mlp'; end   % Asegurar que sea fila


% --- 3.5 Visualización de Muestra Preprocesada e Histograma ---
fprintf('\n--- Visualizaciones Post-Preprocesamiento ---\n');
% Mostrar 5 imágenes de muestra preprocesadas (salida del primer kernel, después de pooling)
try
    display_sample_preprocessed_images_cnn(train_image_paths, 5, chosen_color_filter, img_size, chosen_kernels_list{1}, activation_func, pool_params);
catch ME_vis_sample
    fprintf('Advertencia: No se pudieron mostrar imágenes de muestra preprocesadas: %s\n', ME_vis_sample.message);
end

% Generar Histograma RGB promedio de imágenes de entrenamiento preprocesadas
try
    plot_avg_rgb_histogram_cnn(train_image_paths, chosen_color_filter, img_size);
catch ME_vis_hist
    fprintf('Advertencia: No se pudo generar el histograma RGB: %s\n', ME_vis_hist.message);
end


% --- 4. Configuración de la Red Neuronal FC (usando BP_neuronal_network) ---
fprintf('\n--- Configurando la Red Neuronal (MLP Parte FC) ---\n');
num_flattened_features = size(X_train_mlp, 1);
num_output_neurons = 1; % 1 neurona con sigmoide para clasificación binaria

architecture_mlp = [num_flattened_features, fc_hidden_neurons, num_output_neurons];
fprintf('Arquitectura MLP (FC): [%s]\n', num2str(architecture_mlp));

% Funciones de activación para MLP (usando el formato de string de BP_neuronal_network)
num_mlp_layers_with_activation = length(fc_hidden_neurons) + 1; % Capas ocultas + capa de salida
mlp_activation_functions_str = cell(1, num_mlp_layers_with_activation);
mlp_derivative_functions_str = cell(1, num_mlp_layers_with_activation);

% Para capas ocultas: ReLU
relu_act_str = 'max(0, net)';
relu_deriv_str = 'double(act > 0)'; % Derivada de ReLU respecto a la activación 'act'

% Para capa de salida: Sigmoide
sigmoid_act_str = '1./(1+exp(-net))';
sigmoid_deriv_str = 'act.*(1-act)'; % Derivada de Sigmoide respecto a la activación 'act'

for l_idx = 1:length(fc_hidden_neurons)
    mlp_activation_functions_str{l_idx} = relu_act_str;
    mlp_derivative_functions_str{l_idx} = relu_deriv_str;
end
mlp_activation_functions_str{end} = sigmoid_act_str;
mlp_derivative_functions_str{end} = sigmoid_deriv_str;

% Crear instancia de la red MLP
% NOTA: Asegúrate que la clase BP_neuronal_network.m esté en el path de MATLAB
if exist('BP_neuronal_network', 'class')
    mlp_network = BP_neuronal_network(architecture_mlp, mlp_activation_functions_str, mlp_derivative_functions_str);
else
    error('La clase BP_neuronal_network.m no se encontró. Asegúrate de que esté en el path de MATLAB.');
end
mlp_network.UseGPU = true; % Puedes cambiar a true si tienes GPU y la clase lo soporta bien.

% --- 5. Entrenamiento de la Red MLP ---
% La clase BP_neuronal_network usa MSE como función de pérdida.
% El prompt mencionaba "ej: Entropía Cruzada Binaria". Si BCE es estrictamente necesario,
% la clase BP_neuronal_network necesitaría una modificación en el cálculo de 'deltas{end}'.
% Por ahora, usaremos la clase tal como está (con MSE).
fprintf('\n--- Iniciando Entrenamiento del MLP (Parte FC) ---\n');
fprintf('Función de pérdida (implícita en BP_neuronal_network): Error Cuadrático Medio (MSE/2)\n');

% Crear figura para el gráfico de error
error_figure_handle = figure;
error_axes_handle = axes(error_figure_handle);
title(error_axes_handle, 'Error (MSE/2) vs. Épocas');
xlabel(error_axes_handle, 'Época');
ylabel(error_axes_handle, 'Error (MSE/2)');
grid(error_axes_handle, 'on');

% Para controlar el número máximo de épocas, modificaremos ligeramente cómo llamamos a entrenar
% o asumimos que el 'desired_precision' se alcanza antes del límite interno de la clase.
% La clase BP_neuronal_network tiene un maxEpocas interno. Si necesitas anularlo explícitamente,
% la clase necesitaría un parámetro adicional o una propiedad para max_epochs.
% Por simplicidad, confiamos en 'desired_precision' o en el límite interno.
% Opcional: Si quieres forzar max_epochs, y la clase no lo permite directamente:
% obj.MaxEpochs = max_epochs; % (Necesitarías añadir esta propiedad a la clase)

[mlp_network_trained, epochs_completed, final_mse_error, mse_error_history, ~, ~, ~] = ...
    mlp_network.entrenar(X_train_mlp, Y_train_mlp, alpha, desired_precision, beta, error_axes_handle);
% Nota: La clase BP_neuronal_network utiliza un 'maxEpocas' interno.
% Para usar el 'max_epochs' del usuario, se debería modificar la clase o pasar como parámetro.
% Aquí, asumimos que 'desired_precision' es el criterio principal o el 'max_epochs' del usuario
% se alinea con el comportamiento de la clase. Si no, la clase podría detenerse antes por su límite interno.
% Para un control exacto de max_epochs:
% 1. Modifica la clase para aceptar max_epochs en `entrenar`.
% 2. O, si la clase BP_neuronal_network tiene una propiedad MaxEpochs, ajústala antes de llamar a entrenar.
% Como no podemos modificar la clase aquí, nos basamos en su comportamiento actual.

fprintf('Entrenamiento del MLP finalizado.\n');
fprintf('Épocas completadas: %d\n', epochs_completed);
fprintf('Error MSE/2 final: %f\n', final_mse_error);

% --- 6. Evaluación del Modelo ---
fprintf('\n--- Evaluando el Modelo en el Conjunto de Prueba ---\n');
% Obtener predicciones del MLP (salidas sigmoides)
Y_pred_raw_test = mlp_network_trained.feedforward(X_test_mlp);

% Convertir salidas sigmoides a clases binarias (0 o 1) usando umbral 0.5
Y_pred_binary_test = round(Y_pred_raw_test);

% Calcular Precisión (Accuracy)
accuracy_test = sum(Y_pred_binary_test == Y_test_mlp) / length(Y_test_mlp);
fprintf('Precisión (Accuracy) en el conjunto de prueba: %.2f%%\n', accuracy_test * 100);

% Matriz de Confusión (Opcional, pero útil)
if exist('confusionchart', 'file') % Requiere Statistics and Machine Learning Toolbox o Deep Learning Toolbox
    figure;
    actual_categorical = categorical(Y_test_mlp, [0 1], strrep(species, '_', ' '));
    predicted_categorical = categorical(Y_pred_binary_test, [0 1], strrep(species, '_', ' '));
    confusionchart(actual_categorical, predicted_categorical, ...
        'Title', 'Matriz de Confusión en Conjunto de Prueba', ...
        'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
else
    fprintf('Advertencia: `confusionchart` no disponible. Omitiendo matriz de confusión gráfica.\n');
    fprintf('Calculando matriz de confusión manualmente (0=%s, 1=%s):\n', species{1}, species{2});
    CM = confusionmat(Y_test_mlp, Y_pred_binary_test);
    disp(CM);
    % CM(1,1) = Verdaderos Negativos (clase 0 predicha como 0)
    % CM(1,2) = Falsos Positivos   (clase 0 predicha como 1)
    % CM(2,1) = Falsos Negativos   (clase 1 predicha como 0)
    % CM(2,2) = Verdaderos Positivos (clase 1 predicha como 1)
end

% --- 7. Prueba Interactiva Post-Entrenamiento ---
fprintf('\n--- Modo de Prueba Interactiva ---\n');
num_interactive_tests = 3; % Mínimo 3 según el prompt
for i_test = 1:num_interactive_tests
    fprintf('\nSeleccione imagen de prueba interactiva #%d:\n', i_test);
    [img_filename, img_pathname] = uigetfile(...
        {'*.jpg;*.jpeg;*.png;*.bmp', 'Archivos de Imagen (*.jpg, *.jpeg, *.png, *.bmp)'; '*.*', 'Todos los Archivos (*.*)'}, ...
        'Seleccionar imagen para predicción');
    
    if isequal(img_filename, 0) || isequal(img_pathname, 0)
        fprintf('No se seleccionó archivo. Omitiendo prueba interactiva #%d.\n', i_test);
        continue;
    end
    
    full_img_path = fullfile(img_pathname, img_filename);
    try
        interactive_img_raw = imread(full_img_path);
        
        % Aplicar el MISMO preprocesamiento y extracción de características
        % (Convertimos a cell para usar la misma función)
        X_interactive_mlp = preprocess_and_extract_features_cnn({full_img_path}, chosen_color_filter, img_size, chosen_kernels_list, activation_func, pool_params);
        
        if isempty(X_interactive_mlp)
            fprintf('Error: No se pudieron extraer características de la imagen interactiva.\n');
            continue;
        end

        % Predecir con la red MLP entrenada
        pred_raw_interactive = mlp_network_trained.feedforward(X_interactive_mlp);
        pred_binary_interactive = round(pred_raw_interactive);
        
        predicted_species_name = species{pred_binary_interactive + 1}; % +1 porque los labels son 0,1 y los índices de cell 1,2
        
        % Mostrar imagen y predicción
        figure;
        imshow(interactive_img_raw);
        title_str = sprintf('Imagen: %s\nPredicción: %s (Salida raw: %.4f)', ...
                            strrep(img_filename, '_', '\_'), ...
                            strrep(predicted_species_name, '_', ' '), ...
                            pred_raw_interactive);
        title(title_str);
        
        fprintf('Imagen "%s" - Predicción: %s (Salida raw de la red: %.4f)\n', ...
                img_filename, predicted_species_name, pred_raw_interactive);
                
    catch ME_interactive
        fprintf('Error procesando la imagen interactiva "%s": %s\n', img_filename, ME_interactive.message);
    end
end

fprintf('\n--- Script Finalizado ---\n');


% =========================================================================
% --- Funciones Auxiliares ---
% (Colocar al final del script o en archivos .m separados y añadirlos al path)
% =========================================================================

function kernels = define_kernels_cnn()
    % Define una estructura de kernels convolucionales predefinidos
    kernels.Identity = [0 0 0; 0 1 0; 0 0 0];
    kernels.EdgeDetect_SobelX = [-1 0 1; -2 0 2; -1 0 1]; % Detección de bordes Sobel X
    kernels.EdgeDetect_SobelY = [-1 -2 -1; 0 0 0; 1 2 1]; % Detección de bordes Sobel Y
    kernels.EdgeDetect_Laplacian = [0 1 0; 1 -4 1; 0 1 0]; % Detección de bordes Laplaciano
    kernels.Sharpen = [0 -1 0; -1 5 -1; 0 -1 0]; % Realce/Sharpen
    kernels.GaussianBlur3x3 = (1/16) * [1 2 1; 2 4 2; 1 2 1]; % Desenfoque Gaussiano
    % kernels.BoxBlur = (1/9) * ones(3);
end

function [image_paths, labels_numeric] = load_and_label_images_cnn(dataset_root_path, species_names, labels_mapping)
    % Carga rutas de imágenes y asigna etiquetas numéricas.
    image_paths = {};
    labels_list = [];
    
    for i = 1:length(species_names)
        species_name = species_names{i};
        species_folder = fullfile(dataset_root_path, species_name);
        
        if ~isfolder(species_folder)
            warning('Carpeta no encontrada para la especie %s: %s', species_name, species_folder);
            continue;
        end
        
        img_files = [dir(fullfile(species_folder, '*.jpg')); ...
                     dir(fullfile(species_folder, '*.jpeg')); ...
                     dir(fullfile(species_folder, '*.png'))];
        
        fprintf('Cargando %d imágenes de la especie "%s"...\n', length(img_files), species_name);
        
        for j = 1:length(img_files)
            image_paths{end+1} = fullfile(species_folder, img_files(j).name); %#ok<AGROW>
            labels_list(end+1) = labels_mapping(species_name); %#ok<AGROW>
        end
    end
    if ~isempty(image_paths)
        % Aleatorizar el orden para que el split entrenamiento/prueba sea más robusto
        rand_indices = randperm(length(image_paths));
        image_paths = image_paths(rand_indices);
        labels_numeric = labels_list(rand_indices);
    else
        labels_numeric = [];
    end
end

function img_out = apply_color_filter_cnn(img_in, filter_type_str)
    % Aplica un filtro de color a la imagen.
    % img_in se espera que sea uint8 o double [0,1]
    was_double = isa(img_in, 'double');
    if was_double
        img_uint8 = uint8(img_in * 255); % Convertir a uint8 para algunas operaciones si es necesario
    else
        img_uint8 = img_in;
    end

    switch lower(filter_type_str)
        case 'grayscale'
            if size(img_uint8, 3) == 3
                % Fórmula estándar de luminancia (Y = 0.299R + 0.587G + 0.114B)
                img_gray_double = 0.299*double(img_uint8(:,:,1)) + ...
                                  0.587*double(img_uint8(:,:,2)) + ...
                                  0.114*double(img_uint8(:,:,3));
                img_out_uint8 = uint8(img_gray_double);
                img_out_uint8 = repmat(img_out_uint8, [1,1,3]); % Mantener 3 canales para consistencia
            else
                img_out_uint8 = img_uint8; % Ya es gris o tiene un solo canal
                if size(img_out_uint8,3) == 1
                   img_out_uint8 = repmat(img_out_uint8, [1,1,3]);
                end
            end
        case 'redemphasis'
            img_out_uint8 = img_uint8;
            if size(img_out_uint8,3)==3
                img_out_uint8(:,:,2) = uint8(double(img_out_uint8(:,:,2)) * 0.5); % Atenuar Verde
                img_out_uint8(:,:,3) = uint8(double(img_out_uint8(:,:,3)) * 0.5); % Atenuar Azul
            end
        case 'greenemphasis'
            img_out_uint8 = img_uint8;
             if size(img_out_uint8,3)==3
                img_out_uint8(:,:,1) = uint8(double(img_out_uint8(:,:,1)) * 0.5); % Atenuar Rojo
                img_out_uint8(:,:,3) = uint8(double(img_out_uint8(:,:,3)) * 0.5); % Atenuar Azul
            end
        case 'blueemphasis'
            img_out_uint8 = img_uint8;
            if size(img_out_uint8,3)==3
                img_out_uint8(:,:,1) = uint8(double(img_out_uint8(:,:,1)) * 0.5); % Atenuar Rojo
                img_out_uint8(:,:,2) = uint8(double(img_out_uint8(:,:,2)) * 0.5); % Atenuar Verde
            end
        case 'invertcolor'
            img_out_uint8 = 255 - img_uint8;
        case 'none'
            img_out_uint8 = img_uint8;
        otherwise
            warning('Filtro de color desconocido: %s. No se aplicará filtro.', filter_type_str);
            img_out_uint8 = img_uint8;
    end
    
    if was_double
        img_out = double(img_out_uint8) / 255.0;
    else
        img_out = img_out_uint8;
    end
end

function conv_map = manual_conv2d_cnn(img_channel_double, kernel_matrix)
    % Realiza convolución 2D usando conv2 de MATLAB.
    % img_channel_double: un solo canal de imagen, tipo double.
    % kernel_matrix: el kernel de convolución.
    conv_map = conv2(img_channel_double, kernel_matrix, 'same');
end

function relu_output = relu_manual_cnn(input_data)
    % Implementación manual de la función de activación ReLU.
    relu_output = max(0, input_data);
end

function pooled_output = max_pooling_manual_cnn(feature_map, pool_s, pool_stride)
    % Implementación manual de Max Pooling 2D.
    % feature_map: mapa de características de entrada.
    % pool_s: tamaño del pooling [pool_height, pool_width].
    % pool_stride: stride del pooling (escalar, se asume igual para H y W).
    
    [map_h, map_w] = size(feature_map);
    pool_h = pool_s(1);
    pool_w = pool_s(2);
    
    out_h = floor((map_h - pool_h) / pool_stride) + 1;
    out_w = floor((map_w - pool_w) / pool_stride) + 1;
    
    pooled_output = zeros(out_h, out_w);
    
    for r = 1:out_h
        for c = 1:out_w
            % Calcular índices en el feature_map original
            r_start = (r-1) * pool_stride + 1;
            r_end = r_start + pool_h - 1;
            c_start = (c-1) * pool_stride + 1;
            c_end = c_start + pool_w - 1;
            
            % Extraer la región y aplicar max pooling
            region = feature_map(r_start:r_end, c_start:c_end);
            pooled_output(r,c) = max(region(:));
        end
    end
end

function features_matrix = preprocess_and_extract_features_cnn(image_path_list, color_filt_str, resize_dims, kernels_cell_array, activ_func_handle, pool_prms)
    % Función completa para preprocesar un conjunto de imágenes y extraer características.
    num_images_to_process = length(image_path_list);
    
    % Procesar la primera imagen para determinar el tamaño de las características aplanadas
    if num_images_to_process == 0
        features_matrix = [];
        return;
    end
    
    first_img_path = image_path_list{1};
    img_sample = imread(first_img_path);
    img_filtered_sample = apply_color_filter_cnn(img_sample, color_filt_str);
    img_resized_sample = imresize(img_filtered_sample, resize_dims);
    if size(img_resized_sample,3) == 1 % Si es escala de grises, replicar a 3 canales
        img_resized_sample = repmat(img_resized_sample, [1 1 3]);
    end
    img_double_sample = double(img_resized_sample) / 255.0; % Normalizar a [0,1]

    all_pooled_flattened_sample = [];
    for k_idx = 1:length(kernels_cell_array)
        current_kernel = kernels_cell_array{k_idx};
        % Aplicar kernel a cada canal y luego promediar (o podrías apilarlos)
        conv_outputs_per_channel = zeros(size(img_double_sample,1), size(img_double_sample,2), size(img_double_sample,3));
        for ch = 1:size(img_double_sample,3)
            conv_outputs_per_channel(:,:,ch) = manual_conv2d_cnn(img_double_sample(:,:,ch), current_kernel);
        end
        % Promediar los mapas de características de los canales para este kernel
        conv_map_avg = mean(conv_outputs_per_channel, 3); 
        
        activated_map = activ_func_handle(conv_map_avg);
        pooled_map = max_pooling_manual_cnn(activated_map, pool_prms.size, pool_prms.stride);
        all_pooled_flattened_sample = [all_pooled_flattened_sample; pooled_map(:)]; %#ok<AGROW>
    end
    num_flattened_feats = length(all_pooled_flattened_sample);
    
    % Inicializar matriz para todas las características
    features_matrix = zeros(num_flattened_feats, num_images_to_process);
    
    % Usar parfor si tienes Parallel Computing Toolbox para acelerar
    parfor i = 1:num_images_to_process
        img_path_current = image_path_list{i};
        try
            current_image_raw = imread(img_path_current);
            
            % 1. Filtro de Color
            img_filtered = apply_color_filter_cnn(current_image_raw, color_filt_str);
            
            % 2. Redimensionamiento
            img_resized = imresize(img_filtered, resize_dims);
            if size(img_resized,3) == 1 % Si es escala de grises, replicar a 3 canales
                img_resized = repmat(img_resized, [1 1 3]);
            end
            img_double = double(img_resized) / 255.0; % Normalizar a [0,1]
            
            % 3. Convolución, Activación, Pooling, Aplanado (por cada kernel)
            all_pooled_flattened_for_this_image = [];
            for k_idx = 1:length(kernels_cell_array)
                current_kernel = kernels_cell_array{k_idx};
                
                conv_outputs_per_channel_par = zeros(size(img_double,1), size(img_double,2), size(img_double,3));
                for ch = 1:size(img_double,3) % Aplicar a cada canal
                    conv_outputs_per_channel_par(:,:,ch) = manual_conv2d_cnn(img_double(:,:,ch), current_kernel);
                end
                conv_map_avg_par = mean(conv_outputs_per_channel_par, 3);
                
                activated_map_par = activ_func_handle(conv_map_avg_par);
                pooled_map_par = max_pooling_manual_cnn(activated_map_par, pool_prms.size, pool_prms.stride);
                
                all_pooled_flattened_for_this_image = [all_pooled_flattened_for_this_image; pooled_map_par(:)]; %#ok<AGROWINPARFOR>
            end
            features_matrix(:, i) = all_pooled_flattened_for_this_image;
        catch ME_parfor
             fprintf('Error procesando imagen en parfor: %s. Imagen: %s. Omitiendo.\n', ME_parfor.message, img_path_current);
             % Dejar la columna como ceros o manejar de otra forma
        end
    end
end

function display_sample_preprocessed_images_cnn(image_paths_list, num_to_show, color_filt, resize_dim, kernel_to_vis, act_func, pool_pars)
    % Muestra N imágenes después de aplicar el primer kernel, activación y pooling.
    figure;
    sgtitle(sprintf('Muestra de Imágenes Preprocesadas (1er Kernel: %s, Post-Pool)', inputname(5))); % Trata de obtener nombre del kernel
    
    actual_num_to_show = min(num_to_show, length(image_paths_list));
    
    for i = 1:actual_num_to_show
        img_raw = imread(image_paths_list{i});
        img_filtered = apply_color_filter_cnn(img_raw, color_filt);
        img_resized = imresize(img_filtered, resize_dim);
        if size(img_resized,3) == 1; img_resized = repmat(img_resized, [1 1 3]); end
        img_double = double(img_resized) / 255.0;
        
        % Aplicar solo el kernel_to_vis
        conv_outputs_ch_vis = zeros(size(img_double,1),size(img_double,2),size(img_double,3));
        for ch = 1:size(img_double,3)
            conv_outputs_ch_vis(:,:,ch) = manual_conv2d_cnn(img_double(:,:,ch), kernel_to_vis);
        end
        conv_map_vis = mean(conv_outputs_ch_vis, 3);
        
        activated_map_vis = act_func(conv_map_vis);
        pooled_map_vis = max_pooling_manual_cnn(activated_map_vis, pool_pars.size, pool_pars.stride);
        
        subplot(1, actual_num_to_show, i);
        imshow(pooled_map_vis, []); % Escalar para visualización
        title(sprintf('Muestra %d', i));
    end
    drawnow;
end

function plot_avg_rgb_histogram_cnn(image_paths_list, color_filt, resize_dim)
    % Calcula y muestra un histograma RGB promedio de las imágenes de entrenamiento preprocesadas.
    num_images = length(image_paths_list);
    if num_images == 0; return; end
    
    % Acumuladores para R, G, B
    all_r_values = [];
    all_g_values = [];
    all_b_values = [];
    
    max_images_for_hist = min(num_images, 50); % Limitar para no consumir demasiada memoria/tiempo
    
    fprintf('Calculando histograma RGB promedio de %d imágenes (máx %d)...\n', num_images, max_images_for_hist);
    for i = 1:max_images_for_hist
        img_raw = imread(image_paths_list{i});
        img_filtered = apply_color_filter_cnn(img_raw, color_filt);
        img_resized = imresize(img_filtered, resize_dim); % uint8 or double
        
        if isa(img_resized, 'double') % Convertir a uint8 si es double [0,1]
            img_to_hist = uint8(img_resized * 255);
        else
            img_to_hist = img_resized; % Asumir uint8
        end

        if size(img_to_hist, 3) == 3
            all_r_values = [all_r_values; img_to_hist(:,:,1)]; %#ok<AGROW>
            all_g_values = [all_g_values; img_to_hist(:,:,2)]; %#ok<AGROW>
            all_b_values = [all_b_values; img_to_hist(:,:,3)]; %#ok<AGROW>
        elseif strcmp(lower(color_filt), 'grayscale') || size(img_to_hist,3) == 1
             all_r_values = [all_r_values; img_to_hist(:)]; %#ok<AGROW> % Tratar como R, G y B serán iguales
             all_g_values = [all_g_values; img_to_hist(:)]; %#ok<AGROW>
             all_b_values = [all_b_values; img_to_hist(:)]; %#ok<AGROW>
        end
    end
    
    if isempty(all_r_values) && isempty(all_g_values) && isempty(all_b_values)
        fprintf('No hay datos de color para generar el histograma (¿Dataset solo en escala de grises y filtro "None"?).\n');
        return;
    end

    figure;
    hold on;
    if ~isempty(all_r_values); histogram(all_r_values(:), 256, 'FaceColor', 'r', 'EdgeColor', 'r', 'FaceAlpha', 0.5); end
    if ~isempty(all_g_values); histogram(all_g_values(:), 256, 'FaceColor', 'g', 'EdgeColor', 'g', 'FaceAlpha', 0.5); end
    if ~isempty(all_b_values); histogram(all_b_values(:), 256, 'FaceColor', 'b', 'EdgeColor', 'b', 'FaceAlpha', 0.5); end
    hold off;
    
    title('Histograma RGB Promedio de Imágenes de Entrenamiento Preprocesadas');
    xlabel('Intensidad del Pixel');
    ylabel('Frecuencia');
    legend({'Rojo', 'Verde', 'Azul'});
    grid on;
    drawnow;
end 