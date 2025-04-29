%Realizado por Juan Esteban Fuentes, Laura Latorre y Duvan Santiago Matallana
classdef BP_neuronal_network
    properties
        Arquitectura          % Vector con el número de neuronas por capa [in, hn, out]
        NumCapas              % Número total de capas (incluyendo entrada y salida)
        Pesos                 % Cell array con las matrices de pesos {W1, W2, ...}
        Sesgos                % Cell array con los vectores de sesgos {b1, b2, ...}
        FuncionesActivacion   % Cell array con handles a funciones de activación {@(net)..., ...}
        DerivadasActivacion   % Cell array con handles a funciones derivadas {@(act)..., ...}
    end

    methods
        function obj = BP_neuronal_network(arquitectura, funcActivacionStr, derivActivacionStr)
            %constructor de la clase
            if nargin > 0
                obj.Arquitectura = arquitectura;
                obj.NumCapas = length(arquitectura);
                if length(funcActivacionStr) ~= obj.NumCapas -1 || length(derivActivacionStr) ~= obj.NumCapas -1
                    error('El número de funciones/derivadas debe coincidir con el número de capas ocultas + salida.');
                end
                %inicializar los pesos y sesgos (bias)
                obj.Pesos = cell(1, obj.NumCapas - 1);
                obj.Sesgos = cell(1, obj.NumCapas - 1);
                rng('shuffle'); % Inicializar generador aleatorio
                for i = 1:(obj.NumCapas-1)
                    obj.Pesos{i} = (rand(arquitectura(i+1), arquitectura(i))*2-1);
                    obj.Sesgos{i}=zeros(arquitectura(i+1),1);
                    fprintf('Capa %d: Pesos size [%d, %d], Sesgos size [%d, 1]\n', ...
                            i, size(obj.Pesos{i},1), size(obj.Pesos{i},2), size(obj.Sesgos{i},1));
                end
                %--Cambiar String de función a función handle--
                obj.FuncionesActivacion = cell(1,obj.NumCapas -1);
                obj.DerivadasActivacion = cell(1, obj.NumCapas -1);
                for i = 1:(obj.NumCapas - 1)
                    try
                        obj.FuncionesActivacion{i} = str2func(['@(net) ' funcActivacionStr{i}]);
                        obj.DerivadasActivacion{i} = str2func(['@(act) ' derivActivacionStr{i}]);
                    catch ME
                        error('Error convirtiendo string a función en capa %d: %s\nString Func: %s\nString Deriv: %s', ...
                               i, ME.message, funcActivacionStr{i}, derivActivacionStr{i});
                    end
                end
                disp('Funciones de activación y derivadas convertidas a handles.');
            end
        end

        function obj = setInitialWeights(obj, initialPesos, initialSesgos)
            % Establece pesos y sesgos iniciales DESPUÉS de crear el objeto.
            % Sobrescribe los pesos/sesgos inicializados aleatoriamente.
            % initialPesos, initialSesgos: Cell arrays con el formato correcto.
            
            if isempty(obj.Arquitectura)
                 error('La arquitectura debe estar definida antes de establecer pesos iniciales.');
            end
            if length(initialPesos) ~= obj.NumCapas - 1 || length(initialSesgos) ~= obj.NumCapas - 1
                error('El número de cell arrays de pesos/sesgos iniciales no coincide con la arquitectura.');
            end
            
            % Validar dimensiones de cada matriz/vector (opcional pero recomendado)
            for i = 1:(obj.NumCapas - 1)
                expected_W_size = [obj.Arquitectura(i+1), obj.Arquitectura(i)];
                expected_b_size = [obj.Arquitectura(i+1), 1];
                if ~isequal(size(initialPesos{i}), expected_W_size)
                     error('Tamaño de pesos iniciales para capa %d incorrecto. Esperado [%d %d], recibido [%d %d].', ...
                           i, expected_W_size(1), expected_W_size(2), size(initialPesos{i},1), size(initialPesos{i},2));
                end
                 if ~isequal(size(initialSesgos{i}), expected_b_size)
                      error('Tamaño de sesgos iniciales para capa %d incorrecto. Esperado [%d %d], recibido [%d %d].', ...
                           i, expected_b_size(1), expected_b_size(2), size(initialSesgos{i},1), size(initialSesgos{i},2));
                 end
            end
            
            obj.Pesos = initialPesos;
            obj.Sesgos = initialSesgos;
            disp('Pesos y sesgos iniciales establecidos desde archivo.');
        end
        
        function [salidaFinal, activaciones, nets] = feedforward(obj, entrada)
            % Realiza la pasada hacia adelante (feedforward)
            % 'entrada' debe ser un vector columna
            if size(entrada, 1) ~= obj.Arquitectura(1)
                error('Dimensiones de entrada (filas=%d) no coinciden con la arquitectura (%d).', size(entrada,1), obj.Arquitectura(1));
            end
            if size(entrada, 2) ~= 1
                error('La función feedforward espera un solo patrón.')
            end
            activacion = entrada; % Activación inicial es la entrada
            activaciones = cell(1,obj.NumCapas); % Guardar activación de cada capa
            nets = cell(1,obj.NumCapas - 1); % Guardar entrada neta (z) de cada capa (excepto entrada)
            activaciones{1} = activacion;

            for i = 1:(obj.NumCapas - 1)
                W = obj.Pesos{i};
                b = obj.Sesgos{i};
                funAct = obj.FuncionesActivacion{i};

                %Calcular la entrada neta (suma ponderada Neta)
                nets{i} = W * activacion + b;

                %Calcular la activación de la neurona
                activacion = funAct(nets{i});
                activaciones{i+1} = activacion; % Guardar activación de la capa actual (i+1)
            end
            salidaFinal = activacion; % La última activación es la salida de la red
        end

        function [obj, epoca, errorFinal, historial_error, historial_pesos, historial_sesgos, savedEpochs] = entrenar(obj, X_entrada, Y_deseada, alpha, precision, beta, axesHandleError)
                        % Entrena la red usando backpropagation
            % DEVUELVE: 
            %   obj: El objeto red entrenado
            %   epoca: El número final de épocas realizadas
            %   errorFinal: El error MSE final alcanzado
            %   historial_error: Vector con el MSE de cada época
            %   historial_pesos: Cell array con los pesos de cada época
            %   historial_sesgos: Cell array con los sesgos de cada época
            % X_entrada: matriz de entradas
            % Y_deseada: matriz de salidas deseadas
            
            % --- Validaciones de Dimensiones ---
            if size(X_entrada, 1) ~= obj.Arquitectura(1)
                error('Número de características de entrada no coincide con arquitectura(%d).', size(X_entrada, 1), obj.Arquitectura(1));
            end
            if size(Y_deseada, 1) ~= obj.Arquitectura(end)
                 error('Número de características de salida no coincide con arquitectura (%d).', size(Y_deseada, 1), obj.Arquitectura(end));
            end
            numPatrones = size(X_entrada, 2);
            if size(Y_deseada, 2) ~= numPatrones
                error('Número de patrones de entrada (%d) y salida (%d) no coincide.', numPatrones, size(Y_deseada, 2));
            end

            %Validación de parámetro Beta
            if nargin < 6 % Si Beta no se proporciona
                warning('Coeficiente de Momentum (Beta) no proporcionado. Usando Beta = 0 (sin momentum).');
                beta = 0;
            elseif ~isnumeric(beta) || ~isscalar(beta) || beta < 0 || beta >= 1
                 error('Beta debe ser un escalar numérico entre 0 y 1. Valor recibido: %g', beta);
            end
            if nargin>=7 && isgraphics(axesHandleError)
                hl = animatedline(axesHandleError,'Marker','.','LineStyle','-');
            else
                hl = [];
            end

            epoca = 0;  % Inicializar las épocas
            errorTotal = inf;  % Inicializar error alto
            % --- Inicializar Históricos ---
            historial_error = []; 
            historial_pesos = {}; % Cell array para guardar cell arrays de pesos
            historial_sesgos = {}; % Cell array para guardar cell arrays de sesgos
            savedEpochs = []; 
            
            % --- INICIALIZAR CAMBIOS ANTERIORES (para Momentum) ---
                        delta_W_prev = cell(1, obj.NumCapas - 1);
            delta_b_prev = cell(1, obj.NumCapas - 1);
            for l = 1:(obj.NumCapas - 1)
                delta_W_prev{l} = zeros(size(obj.Pesos{l})); % Inicializar a cero
                delta_b_prev{l} = zeros(size(obj.Sesgos{l})); % Inicializar a cero
            end

            fprintf('\n--- Iniciando Entrenamiento ---\n');
            fprintf('Patrones: %d, Alpha: %.4f, Precisión requerida: %.g\n', numPatrones, alpha, precision);
             maxEpocas = 5000000; % Límite de seguridad

            while errorTotal > precision && epoca < maxEpocas
                epoca = epoca + 1;
                errorActual = 0;
                idx_shuffled = randperm(numPatrones);
                X_entrada_shuffled = X_entrada(:, idx_shuffled);
                Y_deseada_shuffled = Y_deseada(:, idx_shuffled);
                for i = 1:numPatrones
                    x = X_entrada_shuffled(:, i); 
                    Yd = Y_deseada_shuffled(:, i);
                    
                    %Primero se realiza el feedforward
                    [Yo, activaciones, nets] = obj.feedforward(x);

                    %Cálculo del error para el patrón
                    errorPatron = Yd - Yo;
                    errorActual = errorActual + 0.5 .* sum(errorPatron.^2);
                    
                    %Backpropagation - Cálculo de deltas
                    deltas = cell(1,obj.NumCapas - 1);

                    %Delta de la capa de salida
                    activacion_salida = activaciones{end}; %Para la capa L
                    net_salida = nets{end};
                    derivadaSalida = obj.DerivadasActivacion{end}(activacion_salida);
                    deltas{end} = errorPatron .* derivadaSalida;

                    %Deltas de capas ocultas
                    for l = (obj.NumCapas - 2):-1:1  % Para cada capa l = L-1 ... 1
                        W_siguiente = obj.Pesos{l+1}; %w(l+1)
                        delta_siguiente = deltas{l+1}; %δ^(l+1)
                        activacion_l = activaciones{l+1}; % a^(l)
                        derivadaOculta = obj.DerivadasActivacion{l}(activacion_l);
                        errorPropagado = W_siguiente' * delta_siguiente; % (W^(l+1))' * δ^(l+1)
                        deltas{l} = errorPropagado .* derivadaOculta; %δ^(l)
                    end

                    %Actualización de pesos y sesgos
                    for l = 1:(obj.NumCapas - 1) % Para cada capa l (desde la 1ra oculta hasta la salida)
                        activacion_previa = activaciones{l}; % Activación de la capa anterior (a^{l-1})
                        delta_l = deltas{l};                 % Delta calculado para la capa actual δ^(l)

                        % Calcular gradientes
                        grad_W = delta_l * activacion_previa'; % Gradiente pesos W^l: (n_l x 1) * (1 x n_{l-1}) = (n_l x n_{l-1})
                        grad_b = delta_l;                      % Gradiente bias b^l: (n_l x 1)

                        % Calcular el cambio actual debido al gradiente
                        current_gradient_step_W = alpha * grad_W;
                        current_gradient_step_b = alpha * grad_b;

                        % Calcular el cambio total CON momentum
                        % delta(t) = alpha*grad(t) + beta*delta(t-1)
                        delta_W = current_gradient_step_W + beta * delta_W_prev{l};
                        delta_b = current_gradient_step_b + beta * delta_b_prev{l};

                        % Actualizar pesos y sesgos de la capa l
                        obj.Pesos{l} = obj.Pesos{l} + delta_W;
                        obj.Sesgos{l} = obj.Sesgos{l} + delta_b; 

                        % Guardar el CAMBIO TOTAL de esta iteración para usarlo 
                        % como el término de momentum en la SIGUIENTE iteración
                        delta_W_prev{l} = delta_W; 
                        delta_b_prev{l} = delta_b;
                    end
                end
                
                % Calcular error total promedio de la época (recalculando sobre el dataset original)
                % Es más preciso calcular el error sobre todo el dataset después de actualizar pesos
                errorTotal = 0;
                for i_eval = 1:numPatrones
                      Yo_eval = obj.feedforward(X_entrada(:, i_eval)); % Ojo: usa X_entrada original
                      errorTotal = errorTotal + 0.5 * sum((Y_deseada(:, i_eval) - Yo_eval).^2);
                end
                errorTotal = errorTotal / numPatrones; % Promedio MSE/2
                % Guardar en Históricos 
                historial_error(end+1) = errorTotal;
                % Guardar pesos/sesgos (menos frecuente para ahorrar memoria)
                 if mod(epoca, 10) == 0 || epoca == 1 || errorTotal <= precision 
                     if ~isempty(hl) && isvalid(hl)
                         addpoints(hl, epoca, errorTotal); % <-- Añade el punto
                         drawnow limitrate; % <-- Refresca la gráfica animada
                     end
                     historial_pesos{end+1} = obj.Pesos;   
                     historial_sesgos{end+1} = obj.Sesgos;
                     savedEpochs(end+1) = epoca; 
                 end

                % Imprimir progreso
                if mod(epoca, 250) == 0 || errorTotal <= precision || epoca == 1
                  fprintf('Época: %d, Error MSE/2: %.8f\n', epoca, errorTotal);
                end
                % Condición de seguridad
                if epoca > 50 && historial_error(epoca) > historial_error(epoca-10)
                    break
                end
            end
            if epoca == maxEpocas
                 warning('Entrenamiento detenido: Límite máximo de épocas (%d) alcanzado.', maxEpocas);
            end
            errorFinal = errorTotal; % Guardar el último error calculado
            fprintf('\n--- Entrenamiento Finalizado ---\n');
            fprintf('Épocas totales: %d\n', epoca);
            fprintf('Error final MSE/2: %.8f\n', errorFinal);
        end

        function guardarPesos(obj, filepath) % CAMBIADO nombre de this a obj
             if isempty(obj.Pesos) || isempty(obj.Sesgos); warning('Nada que guardar.'); return; end
             try
                 fid = fopen(filepath, 'wt'); 
                 if fid == -1; error('No se pudo abrir "%s" para guardar.', filepath); end
                 
                 % 1. Guardar Arquitectura (como string separado por espacios)
                 fprintf(fid, '%d ', obj.Arquitectura); 
                 fprintf(fid, '\n'); % Nueva línea después de la arquitectura
                 
                 % 2. Guardar Pesos y Sesgos (un número por línea)
                 for l = 1:(obj.NumCapas - 1)
                     W = obj.Pesos{l};  % Dim: [out x in]
                     b = obj.Sesgos{l}; % Dim: [out x 1]
                     
                     % Escribir W elemento por elemento (MATLAB lee por columnas con fscanf)
                     % Para que reshape funcione bien al leer, guardamos por columnas (orden de MATLAB)
                     for col = 1:size(W, 2) % Iterar por columnas (entrada)
                         for row = 1:size(W, 1) % Iterar por filas (salida)
                             fprintf(fid, '%.15g\n', W(row, col)); 
                         end
                     end
                     
                     % Escribir b elemento por elemento
                     for row = 1:length(b)
                          fprintf(fid, '%.15g\n', b(row));
                     end
                 end
                 fclose(fid);
                 fprintf('Pesos guardados correctamente en "%s".\n', filepath); % Mensaje consola opcional
             catch ME
                 warning('Error guardando pesos en %s: %s', filepath, ME.message);
                 if exist('fid','var') && fid ~= -1; try fclose(fid); catch; end; end 
             end
        end % Fin guardarPesos

        % --- MÉTODO cargarPesos (CORREGIDO y devuelve arquitectura) ---
        function [obj, arquitectura] = cargarPesos(obj, filepath) % CAMBIADO nombre de this a obj, devuelve arch
             if ~isfile(filepath); error('Archivo pesos "%s" no encontrado.', filepath); end
             fid = fopen(filepath, 'r');
             if fid == -1; error('No se pudo abrir el archivo de pesos.'); end
             
             try
                 % Leer arquitectura
                 header = fgetl(fid);
                 arquitectura = str2num(header); %#ok<ST2NM>
                 if isempty(arquitectura) || ~isvector(arquitectura) || any(arquitectura<=0); error('No se pudo leer arquitectura válida de "%s".', filepath); end
                 
                 % Leer el resto como números
                 data = fscanf(fid, '%f'); % Lee todos los números restantes en una columna
                 fclose(fid); % Cerrar aquí después de leer todo
                 
                 if isempty(data); error('No se encontraron datos numéricos de pesos/sesgos en "%s".', filepath); end
                 
                 % Configurar objeto con arquitectura leída
                 obj.Arquitectura = arquitectura;
                 obj.NumCapas = numel(arquitectura);
                 obj.Pesos = cell(1, obj.NumCapas - 1);
                 obj.Sesgos = cell(1, obj.NumCapas - 1);
                 
                 % Reconstruir Pesos y Sesgos
                 idx = 1; % Puntero para recorrer 'data'
                 for l = 1:(obj.NumCapas-1)
                     nOut = arquitectura(l+1); % Neuronas en capa actual (l+1) -> Filas de W, b
                     nIn = arquitectura(l);   % Neuronas en capa anterior (l) -> Columnas de W
                     
                     totalW = nOut * nIn;
                     totalB = nOut;
                     
                     % Validar si hay suficientes datos restantes
                     if idx + totalW - 1 > numel(data)
                         error('Datos insuficientes en "%s" para leer pesos W%d (se esperaban %d).', filepath, l, totalW);
                     end
                     % --- CORRECCIÓN RESHAPE PESOS: [nOut, nIn] ---
                     % Como guardamos por columnas, reshape llena por columnas, resultando en [nOut x nIn]
                     obj.Pesos{l} = reshape(data(idx : idx + totalW - 1), [nOut, nIn]);
                     idx = idx + totalW;
                     
                     if idx + totalB - 1 > numel(data)
                          error('Datos insuficientes en "%s" para leer sesgos b%d (se esperaban %d).', filepath, l, totalB);
                     end
                      % --- CORRECCIÓN RESHAPE SESGOS: [nOut, 1] ---
                     obj.Sesgos{l} = reshape(data(idx : idx + totalB - 1), [nOut, 1]);
                     idx = idx + totalB;
                 end
                 
                 if idx - 1 ~= numel(data)
                      warning('Es posible que sobren datos en el archivo de pesos "%s".', filepath);
                 end

                 fprintf('Pesos y sesgos cargados desde "%s".\n', filepath); % Mensaje consola opcional

             catch ME
                 if exist('fid','var') && fid ~= -1; try fclose(fid); catch; end; end 
                 obj.Pesos = {}; obj.Sesgos = {}; % Limpiar en caso de error
                 fprintf('Error al cargar pesos desde %s: %s\n', filepath, ME.message);
                 rethrow(ME); 
             end
        end

        function [ConfMatPercent, ConfMatCounts] = evaluarClasificacion(obj, X_eval, Y_eval)
            % Evalúa el rendimiento de clasificación implícito de la red.
            % Compara la salida de la red para cada entrada X_eval(:,i) 
            % con todas las salidas deseadas Y_eval(:,j) para determinar
            % cuál es la más cercana ("clasificación").
            % Devuelve una matriz de confusión en porcentajes (filas suman 100%)
            % y la matriz de confusión con conteos absolutos.
            %
            % X_eval: Matriz de entradas para evaluación (num_entradas x num_patrones)
            % Y_eval: Matriz de salidas deseadas (num_salidas x num_patrones)

            if isempty(obj.Pesos) || isempty(obj.Sesgos); error('La red no parece estar inicializada o entrenada.'); end
             if size(X_eval, 2) ~= size(Y_eval, 2); error('El número de patrones de entrada y salida para evaluación no coincide.'); end
             if size(X_eval, 1) ~= obj.Arquitectura(1) || size(Y_eval, 1) ~= obj.Arquitectura(end); error('Las dimensiones de X_eval o Y_eval no coinciden con la arquitectura de la red.'); end
             numPatronesEval = size(X_eval, 2);
             ConfMatCounts = zeros(numPatronesEval, numPatronesEval); 
             fprintf('Evaluando clasificación...\n');
             for i = 1:numPatronesEval 
                 x_i = X_eval(:, i); Yo_i = obj.feedforward(x_i); 
                 distancias = sum((Yo_i - Y_eval).^2, 1); % Distancia a todos los targets Y_eval
                 [~, k_pred] = min(distancias); 
                 ConfMatCounts(i, k_pred) = ConfMatCounts(i, k_pred) + 1;
                 if mod(i, 50) == 0; fprintf('Evaluado patrón %d de %d\n', i, numPatronesEval); end
             end
             fprintf('Evaluación completada.\n');
             ConfMatPercent = zeros(size(ConfMatCounts)); sumRows = sum(ConfMatCounts, 2); 
             nonZeroRows = sumRows > 0;
             ConfMatPercent(nonZeroRows, :) = (ConfMatCounts(nonZeroRows, :) ./ sumRows(nonZeroRows)) * 100;
             ConfMatPercent(~nonZeroRows, :) = NaN; % Marcar filas no evaluadas
             if any(~nonZeroRows); warning('Algunos patrones de entrada no tuvieron instancias en la matriz de confusión.'); end
             % Quitar los disp para no llenar la consola desde la App
             % disp('Matriz de Confusión (Conteos):'); disp(ConfMatCounts);
             % disp('Matriz de Confusión (Porcentajes %):'); disp(ConfMatPercent);
        end

    end

    methods (Static)

        function config = leerConfig(filepath)
            % ... (código leerConfig que lee alpha, beta, precision, arquitectura) ...
             config = struct(); if ~isfile(filepath); warning('Archivo config "%s" no encontrado.', filepath); return; end
             try; lines = readlines(filepath, "Encoding", "UTF-8"); catch ME; warning('No se pudo leer config "%s": %s.', filepath, ME.message); return; end
             fprintf('Leyendo configuración desde "%s"...\n', filepath); keysFound = {};
             for i = 1:length(lines); line = strtrim(lines{i}); if isempty(line) || startsWith(line, '#'); continue; end; parts = split(line, '='); if length(parts) < 2; warning('Línea mal formada: "%s". Ignorando.', line); continue; end; key = lower(strtrim(parts{1})); valueStr = strtrim(strjoin(parts(2:end), '='));
                 try; switch key; case 'arquitectura'; val = str2num(valueStr); if ~isempty(val) && isnumeric(val) && isvector(val) && all(val > 0) && all(fix(val)==val); config.arquitectura = val; keysFound{end+1} = key; else; warning('Valor inválido para arquitectura: "%s".', valueStr); end; case 'alpha'; val = str2double(valueStr); if ~isnan(val) && val > 0; config.alpha = val; keysFound{end+1} = key; else; warning('Valor inválido para alpha: "%s".', valueStr); end; case 'precision'; val = str2double(valueStr); if ~isnan(val) && val > 0; config.precision = val; keysFound{end+1} = key; else; warning('Valor inválido para precision: "%s".', valueStr); end; case 'beta'; val = str2double(valueStr); if ~isnan(val) && val >= 0 && val < 1; config.beta = val; keysFound{end+1} = key; else; warning('Valor inválido para beta: "%s".', valueStr); end; otherwise; warning('Clave desconocida "%s". Ignorando.', key); end; catch ME; warning('Error procesando clave "%s": %s.', key, ME.message); end; end
             if ~isempty(fieldnames(config)); fprintf('Configuración cargada (%s).\n', strjoin(keysFound, ', ')); else; warning('No se cargó configuración válida desde "%s".', filepath); end
        end

        function funciones = leerFunciones(filepath)
             % ... (código leerFunciones sin cambios) ...
             funciones = {}; fid = fopen(filepath, 'r'); if fid == -1; error('No se pudo abrir el archivo de funciones: %s', filepath); end
             while ~feof(fid); line = strtrim(fgetl(fid)); if ~isempty(line) && ~startsWith(line, '#'); funciones{end+1, 1} = line; end; end; fclose(fid);
             fprintf('Funciones cargadas desde ''%s'': %d funciones.\n', filepath, length(funciones));
        end

        function guardarConfig(filepath, config_struct)
            % Guarda la configuración relevante en un archivo.
            % Sobrescribe el archivo si existe.
             fprintf('Intentando guardar configuración en "%s"...', filepath); % Mensaje inicio
             fid = fopen(filepath, 'wt'); % Usar 'wt' para escribir texto
             if fid == -1
                 warning('No se pudo abrir "%s" para escribir la configuración.', filepath);
                 return;
             end
             fprintf(fid, '# Última configuración de entrenamiento utilizada (%s)\n', datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
             
             % Guardar parámetros si existen en la estructura
             if isfield(config_struct, 'alpha')
                 fprintf(fid, 'alpha = %.5g\n', config_struct.alpha);
             end
              if isfield(config_struct, 'beta') % Añadido Beta
                 fprintf(fid, 'beta = %.5g\n', config_struct.beta);
             end
             if isfield(config_struct, 'precision')
                 fprintf(fid, 'precision = %.5g\n', config_struct.precision);
             end
             if isfield(config_struct, 'arquitectura') % Cambiado nombre de clave y formato
                  % Usar mat2str para formato [N M P] o sprintf
                  arch_str = sprintf('%d ', config_struct.arquitectura); 
                  fprintf(fid, 'arquitectura = [%s]\n', strtrim(arch_str)); % Formato [N M P]
             end
             
             fclose(fid);
             fprintf(' Configuración guardada.\n'); % Mensaje fin
        end
    end
end