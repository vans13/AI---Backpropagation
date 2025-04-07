%Realizado por Juan Esteban Fuentes, Laura Latorre y Duvan Santiago Matallana
clear; close all; clc;
% --- Valores Predeterminados (Intentar leer de config.txt) ---
configFile = 'config.txt';
config_def = struct();
% Establecer defaults genéricos por si falla la lectura
alpha_def = 0.1;          
precision_def = 0.001;    
arquitectura_def = [2, 3, 1]; 
config_def.alpha = alpha_def; % Asegurar que los campos existan con nombres correctos
config_def.precision = precision_def;
config_def.arquitectura = arquitectura_def;
try
    config_read = BP_neuronal_network.leerConfig(configFile); % Usa tu función readConfig
    if ~isempty(fieldnames(config_read))
        if isfield(config_read, 'alpha'), config_def.alpha = config_read.alpha; end
        if isfield(config_read, 'precision'), config_def.precision = config_read.precision; end
        if isfield(config_read, 'arquitectura'), config_def.arquitectura = config_read.arquitectura; end
        fprintf('Valores base leídos de "%s".\n', configFile);
    else
         fprintf('No se encontró o leyó config válida de "%s". Usando defaults genéricos.\n', configFile);
    end
catch ME 
    fprintf('Error leyendo "%s": %s\n', configFile, ME.message);
    fprintf('Se usarán defaults genéricos.\n');
end
% --- Usar inputdlg para la configuración ---
prompt = {
    'Alfa (tasa de aprendizaje):', ...
    'Precisión(Error objetivo):', ...
    'Arquitectura (ej: [2 5 1]):'
    };
dlgtitle = 'Configuración de Entrenamiento';
dims = [1 70]; % Tamaño de los campos de entrada [líneas anchura]

% Convertir los valores por defecto a strings para mostrarlos
definput = {
    num2str(config_def.alpha, '%.5g'), ... % num2str con formato
    num2str(config_def.precision, '%.5g'), ... % num2str con formato
    mat2str(config_def.arquitectura) % mat2str para arrays
    };

% Mostrar el cuadro de diálogo
answer = inputdlg(prompt, dlgtitle, dims, definput);

% --- Procesar la respuesta del diálogo ---
if isempty(answer)
    % El usuario presionó Cancelar o cerró la ventana
    fprintf('Configuración cancelada por el usuario.\n');
    return; % Terminar el script
else
    % El usuario presionó OK, intentar convertir y validar
    try
        % Convertir respuestas (vienen como strings en un cell array)
        final_alfa = str2double(answer{1});
        final_precision = str2double(answer{2});
        final_arquitectura = str2num(answer{3});

        % --- Validación de los valores ingresados ---
        if isnan(final_alfa) || ~isscalar(final_alfa) || final_alfa <= 0
            error('Valor inválido para Alfa: Debe ser un número positivo.');
        end
        if isnan(final_precision) || ~isscalar(final_precision) || final_precision <= 0
            error('Valor inválido para precisión: Debe ser un número positivo.');
        end
         if isempty(final_arquitectura) || ~isnumeric(final_arquitectura) || ~isrow(final_arquitectura) || length(final_arquitectura) < 2 || any(final_arquitectura <= 0) || any(mod(final_arquitectura, 1) ~= 0)
             error('Valor inválido para Arquitectura: Debe ser un vector fila de al menos 2 enteros positivos (ej., [2 5 1]).');
         end

        % --- Si todo es válido, crear la estructura final ---
        config_final = struct();
        config_final.alfa = final_alfa;
        config_final.precision = final_precision;
        config_final.arquitectura = final_arquitectura;
        fprintf('Configuración aceptada por el usuario:\n');
        disp(config_final);
    catch ME
        % Si hubo error en conversión o validación, mostrar error y salir
        errordlg(sprintf('Error en la configuración ingresada:\n%s\n\nPor favor, ejecute el script de nuevo.', ME.message), 'Error de Configuración');
        return; % Terminar el script
    end
end
% --- Fin del procesamiento de inputdlg ---

% --- Usar config_final para el resto del script ---
alpha = config_final.alfa;
precision = config_final.precision; 
arquitectura = config_final.arquitectura;
%Leer datos usando los nombres de archivo de config_final
try
    X_data = readmatrix('entradas.txt');
    Y_data = readmatrix('salidas.txt');
    % ... (resto de la carga y validación como antes, usando 'arquitectura') ...
catch ME
     fprintf('Error leyendo archivos de datos (%s, %s): %s\n', x_filename, y_filename, ME.message);
     return;
end
%Validar Datos vs Arquitectura ---
try
    if size(X_data, 2) ~= arquitectura(1)
        error('Las columnas de entrada (%d) en "%s" no coinciden con la primera capa de la arquitectura (%d).', size(X_data, 2), x_filename, arquitectura(1));
    end
    if size(Y_data, 2) ~= arquitectura(end)
         error('Las columnas de salida (%d) en "%s" no coinciden con la última capa de la arquitectura (%d).', size(Y_data, 2), y_filename, arquitectura(end));
    end
    if size(X_data, 1) ~= size(Y_data, 1)
        error('El número de filas (patrones) en "%s" y "%s" no coincide.', x_filename, y_filename);
    end
catch ME
     fprintf('Error de validación: %s\n', ME.message);
     clear ME;
     return;
end
% --- Cargar Funciones de Activación/Derivadas ---
try
    funciones_activacion_str = BP_neuronal_network.leerFunciones('funciones_activacion.txt');
    derivadas_activacion_str = BP_neuronal_network.leerFunciones('derivadas_activacion.txt');
    % Validar número de funciones vs arquitectura
    num_capas_a_activar = length(arquitectura) - 1;
    if length(funciones_activacion_str) ~= num_capas_a_activar || ...
       length(derivadas_activacion_str) ~= num_capas_a_activar
        error(['El número de funciones/derivadas leídas (%d/%d) no coincide '...
               'con el número de capas a activar (%d) según la arquitectura [%s].'],...
               length(funciones_activacion_str), length(derivadas_activacion_str), ...
               num_capas_a_activar, num2str(arquitectura));
    end
catch ME
    fprintf('Error leyendo o validando archivos de funciones: %s\n', ME.message);
    clear ME;
    return;
end

% --- Crear la Instancia de la Red ---
try
    red = BP_neuronal_network(arquitectura, funciones_activacion_str, derivadas_activacion_str);
catch ME
     fprintf('Error creando la instancia de la red: %s\n', ME.message);
     clear ME;
    return;
end

% --- Entrenar la Red ---
try
    % Pasar los parámetros finales y los datos cargados
    [red, epocas_realizadas, error_final_entrenamiento, hist_error, hist_pesos, hist_sesgos] = ...
        red.entrenar(X_data, Y_data, alpha, precision);
catch ME
    fprintf('Error durante el entrenamiento: %s\n', ME.message);
    fprintf('Línea del error: %d\n', ME.stack(1).line);
    clear ME;
    return;
end
% --- MOSTRAR RESULTADOS FINALES Y USAR HISTÓRICOS ---
fprintf('\nPesos Finales (W):\n');
for i = 1:length(red.Pesos)
    fprintf('  Pesos Capa %d (Conexión %d -> %d):\n', i, i-1, i); % Capa 0 es entrada
    disp(red.Pesos{i});
end

fprintf('\nBias Finales (b):\n');
for i = 1:length(red.Sesgos)
     fprintf('  Bias Capa %d:\n', i); % Bias para capa i (activada por conexión i)
     disp(red.Sesgos{i});
end

% --- Graficar Histórico de Error ---
if ~isempty(hist_error)
    try
        figure; % Crea una nueva figura
        plot(1:epocas_realizadas, hist_error, '-b', 'LineWidth', 1.5);
        title('Histórico del Error MSE por Época');
        xlabel('Época');
        ylabel('Error Cuadrático Medio (MSE)');
        grid on;
        set(gca, 'YScale', 'log'); % Escala logarítmica puede ser útil para ver el error inicial
        fprintf('\nMostrando gráfico del histórico de error...\n');
    catch ME_plot
        warning('PlottingError:Hist', 'No se pudo generar el gráfico de error. Detalles abajo:'); 
        fprintf(2, 'Error capturado al intentar graficar:\n%s\n', getReport(ME_plot, 'basic')); 
    end
else
    fprintf('\nNo hay datos históricos de error para graficar.\n');
end

% --- Analizar Histórico de Pesos/Sesgos ---
% El histórico de pesos/sesgos puede ser muy grande. 
% Aquí solo mostramos cómo acceder al estado inicial y final como ejemplo.
if epocas_realizadas > 0 && ~isempty(hist_pesos) && ~isempty(hist_sesgos)
   fprintf('\n--- Ejemplo de Históricos Guardados ---\n');
   fprintf('Pesos Iniciales (Época 1):\n');
   disp(hist_pesos{1}); % Muestra los pesos de la primera época
   fprintf('Sesgos Iniciales (Época 1):\n');
   disp(hist_sesgos{1}); % Muestra los sesgos de la primera época
   
   fprintf('\nPesos Finales (Época %d) - (desde histórico):\n', epocas_realizadas);
   disp(hist_pesos{end}); % Muestra los pesos de la última época guardada
   fprintf('Sesgos Finales (Época %d) - (desde histórico):\n', epocas_realizadas);
   disp(hist_sesgos{end}); % Muestra los sesgos de la última época guardada
else
   fprintf('\nNo hay datos históricos de pesos/sesgos para mostrar.\n');
end
fprintf('\n--- Proceso completado ---\n');