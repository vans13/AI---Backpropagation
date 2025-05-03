% Realizado por Juan Esteban Fuentes, Laura Latorre y Duvan Santiago Matallana
classdef BP_neuronal_network
    properties
        Arquitectura          % Vector [in, h1, ..., hn, out]
        NumCapas              % Número total de capas
        Pesos                 % Cell array {W1, W2, ...}
        Sesgos                % Cell array {b1, b2, ...}
        FuncionesActivacion   % Cell array {@(net)..., ...}
        DerivadasActivacion   % Cell array {@(act)..., ...}
        UseGPU logical = false % Flag para usar GPU si está disponible
    end

    methods
        % Constructor de la clase
        function obj = BP_neuronal_network(arquitectura, funcActivacionStr, derivActivacionStr)
            if nargin > 0
                obj.Arquitectura = arquitectura;
                obj.NumCapas = length(arquitectura);
                if length(funcActivacionStr) ~= obj.NumCapas - 1 || length(derivActivacionStr) ~= obj.NumCapas - 1
                    error('Número de funciones/derivadas debe coincidir con capas ocultas + salida.');
                end

                obj.Pesos = cell(1, obj.NumCapas - 1);
                obj.Sesgos = cell(1, obj.NumCapas - 1);
                rng('shuffle'); % Reiniciar generador aleatorio

                for i = 1:(obj.NumCapas-1)
                    % Inicialización de pesos con randn (más estable)
                    obj.Pesos{i} = randn(arquitectura(i+1), arquitectura(i)) * 0.1;
                    obj.Sesgos{i} = zeros(arquitectura(i+1), 1);
                    fprintf('Capa %d: Pesos size [%d, %d], Sesgos size [%d, 1]\n', ...
                            i, size(obj.Pesos{i},1), size(obj.Pesos{i},2), size(obj.Sesgos{i},1));
                end

                obj.FuncionesActivacion = cell(1, obj.NumCapas - 1);
                obj.DerivadasActivacion = cell(1, obj.NumCapas - 1);
                for i = 1:(obj.NumCapas - 1)
                    try
                        obj.FuncionesActivacion{i} = str2func(['@(net) ' funcActivacionStr{i}]);
                        obj.DerivadasActivacion{i} = str2func(['@(act) ' derivActivacionStr{i}]);
                    catch ME
                        error(sprintf('Error convirtiendo string a función (Capa %d):\nFunc: %s\nDeriv: %s\nError: %s', ...
                               i+1, funcActivacionStr{i}, derivActivacionStr{i}, ME.message));
                    end
                end
                disp('Red inicializada. Funciones convertidas a handles.');
            end
        end

        % Establecer pesos y sesgos iniciales (útil para cargar)
        function obj = setInitialWeights(obj, initialPesos, initialSesgos)
            if isempty(obj.Arquitectura); error('Arquitectura no definida para setInitialWeights.'); end
            if length(initialPesos) ~= obj.NumCapas - 1 || length(initialSesgos) ~= obj.NumCapas - 1
                error('Número de cell arrays de pesos/sesgos iniciales no coincide.');
            end
            for i = 1:(obj.NumCapas - 1)
                expected_W_size = [obj.Arquitectura(i+1), obj.Arquitectura(i)];
                expected_b_size = [obj.Arquitectura(i+1), 1];
                if ~isequal(size(initialPesos{i}), expected_W_size)
                     error(sprintf('Tamaño pesos iniciales capa %d incorrecto. Esperado [%d %d], recibido [%d %d].', ...
                           i, expected_W_size(1), expected_W_size(2), size(initialPesos{i},1), size(initialPesos{i},2)));
                end
                 if ~isequal(size(initialSesgos{i}), expected_b_size)
                      error(sprintf('Tamaño sesgos iniciales capa %d incorrecto. Esperado [%d %d], recibido [%d %d].', ...
                            i, expected_b_size(1), expected_b_size(2), size(initialSesgos{i},1), size(initialSesgos{i},2)));
                 end
            end
            obj.Pesos = initialPesos;
            obj.Sesgos = initialSesgos;
            disp('Pesos y sesgos iniciales establecidos externamente.');
        end

        % Feedforward Vectorizado (acepta matriz de entrada)
        function [salidaFinal, activaciones, nets] = feedforward(obj, X_entrada_batch)
            % X_entrada_batch: Matriz de entrada [NumCaracteristicas x NumPatronesBatch]
            if size(X_entrada_batch, 1) ~= obj.Arquitectura(1)
                error(sprintf('Feedforward: Dimensiones entrada (filas=%d) no coinciden con arquitectura (%d).', size(X_entrada_batch,1), obj.Arquitectura(1)));
            end

            activacion = X_entrada_batch;
            activaciones = cell(1, obj.NumCapas);
            nets = cell(1, obj.NumCapas - 1);
            activaciones{1} = activacion;

            for i = 1:(obj.NumCapas - 1)
                W = obj.Pesos{i};
                b = obj.Sesgos{i};
                funAct = obj.FuncionesActivacion{i};

                nets{i} = W * activacion + b; % Suma ponderada + bias (usa expansión implícita)
                % if any(isinf(nets{i}(:)),'all') || any(isnan(nets{i}(:)),'all'); fprintf(2,'>>> DEBUG: Inf/NaN NETS capa %d!\n', i); end

                activacion = funAct(nets{i}); % Aplicar activación element-wise
                % if any(isinf(activacion(:)),'all') || any(isnan(activacion(:)),'all'); fprintf(2,'>>> DEBUG: Inf/NaN ACTIVACION capa %d!\n', i+1); end

                activaciones{i+1} = activacion;
            end
            salidaFinal = activacion; % Salida final para todo el batch
        end

        % Entrenamiento Vectorizado (Batch Completo) con opción GPU
        function [obj, epoca, errorFinal, historial_error, historial_pesos, historial_sesgos, savedEpochs] = entrenar(obj, X_entrada, Y_deseada, alpha, precision, beta, axesHandleError)
            % Validaciones
            if size(X_entrada, 1) ~= obj.Arquitectura(1)
                error(sprintf('Entrenar: Caract. entrada (%d) != Arq. entrada (%d).', size(X_entrada, 1), obj.Arquitectura(1)));
            end
            if size(Y_deseada, 1) ~= obj.Arquitectura(end)
                 error(sprintf('Entrenar: Salidas deseadas (%d) != Arq. salida (%d).', size(Y_deseada, 1), obj.Arquitectura(end)));
            end
            numPatrones = size(X_entrada, 2);
            if size(Y_deseada, 2) ~= numPatrones
                error(sprintf('Entrenar: #Patrones entrada (%d) != #Patrones salida (%d).', numPatrones, size(Y_deseada, 2)));
            end
            if nargin < 6 || isempty(beta); beta = 0; elseif ~isscalar(beta) || beta<0 || beta>=1; error('Beta debe ser escalar [0, 1).'); end

            % Setup GPU
            useGPU = obj.UseGPU && (gpuDeviceCount > 0);
            if useGPU
                fprintf('GPU detectada. Transfiriendo datos a GPU...\n');
                try
                    X_entrada = gpuArray(X_entrada);
                    Y_deseada = gpuArray(Y_deseada);
                    for l = 1:(obj.NumCapas - 1); obj.Pesos{l} = gpuArray(obj.Pesos{l}); obj.Sesgos{l} = gpuArray(obj.Sesgos{l}); end
                    fprintf('Datos transferidos a GPU.\n');
                catch ME_gpu
                    warning('Error transfiriendo a GPU: %s. Usando CPU.', ME_gpu.message); useGPU = false;
                    for l = 1:(obj.NumCapas - 1); obj.Pesos{l} = gather(obj.Pesos{l}); obj.Sesgos{l} = gather(obj.Sesgos{l}); end % Asegurar CPU si falló
                end
            else; fprintf('GPU no disponible o desactivada. Usando CPU.\n'); end

            % Setup Plotting
            hl = []; % Handle para línea animada
            if nargin >= 7 && isgraphics(axesHandleError)
                cla(axesHandleError); hl = animatedline(axesHandleError,'Marker','.','LineStyle','-');
                title(axesHandleError, 'Error MSE/2 vs Épocas'); xlabel(axesHandleError, 'Época'); ylabel(axesHandleError, 'Error (MSE/2)'); grid(axesHandleError,'on'); axesHandleError.YScale = 'log';
            end

            % Inicializaciones
            epoca = 0; errorTotal = inf; maxEpocas = 5000000; % Límite
            historial_error = []; historial_pesos = {}; historial_sesgos = {}; savedEpochs = [];
            delta_W_prev = cell(1, obj.NumCapas - 1); delta_b_prev = cell(1, obj.NumCapas - 1);

            % Inicializar Momentum (en CPU o GPU según 'useGPU')
            for l = 1:(obj.NumCapas - 1)
                zeros_w = zeros(size(obj.Pesos{l}));
                zeros_b = zeros(size(obj.Sesgos{l}));
                if useGPU; delta_W_prev{l} = gpuArray(zeros_w); delta_b_prev{l} = gpuArray(zeros_b);
                else; delta_W_prev{l} = zeros_w; delta_b_prev{l} = zeros_b; end
            end

            fprintf('\n--- Iniciando Entrenamiento Vectorizado ---\n');
            if useGPU
                destinoEntrenamiento = 'GPU';
            else
                destinoEntrenamiento = 'CPU';
            end
            fprintf('Patrones: %d, Alpha: %.4f, Precisión: %.g, Beta: %.4f, Destino: %s\n', numPatrones, alpha, precision, beta, destinoEntrenamiento);

            % Bucle Principal por Épocas (Vectorizado)
            while errorTotal > precision && epoca < maxEpocas
                epoca = epoca + 1;

                % Feedforward para TODO el dataset
                [Y_salida, activaciones, ~] = obj.feedforward(X_entrada); % Usa X_entrada (CPU o GPU)
                if any(isnan(Y_salida(:)),'all') || any(isinf(Y_salida(:)),'all'); fprintf(2,'>>> ERROR FATAL: NaN/Inf en Y_salida epoca %d! Entrenamiento detenido.\n',epoca); break; end

                % Cálculo del Error para TODO el dataset
                Error = Y_deseada - Y_salida; % Matriz de error (CPU o GPU)
                errorTotal = 0.5 * sum(Error.^2, 'all') / numPatrones;
                if isnan(errorTotal) || isinf(errorTotal); fprintf(2,'>>> ERROR FATAL: NaN/Inf en errorTotal epoca %d! Entrenamiento detenido.\n',epoca); break; end

                % Backpropagation Vectorizado
                deltas = cell(1, obj.NumCapas - 1);
                activacion_salida = activaciones{end};
                derivadaSalida = obj.DerivadasActivacion{end}(activacion_salida);
                deltas{end} = Error .* derivadaSalida;

                for l = (obj.NumCapas - 2):-1:1
                    W_siguiente = obj.Pesos{l+1}; delta_siguiente = deltas{l+1};
                    activacion_l = activaciones{l+1}; derivadaOculta = obj.DerivadasActivacion{l}(activacion_l);
                    errorPropagado = W_siguiente' * delta_siguiente;
                    deltas{l} = errorPropagado .* derivadaOculta;
                end

                % Actualización de Pesos y Sesgos (Vectorizado)
                for l = 1:(obj.NumCapas - 1)
                    activacion_previa = activaciones{l}; delta_l = deltas{l};
                    grad_W = (delta_l * activacion_previa') / numPatrones;
                    grad_b = sum(delta_l, 2) / numPatrones;

                    delta_W = (alpha * grad_W) + (beta * delta_W_prev{l});
                    delta_b = (alpha * grad_b) + (beta * delta_b_prev{l});

                    obj.Pesos{l} = obj.Pesos{l} + delta_W;
                    obj.Sesgos{l} = obj.Sesgos{l} + delta_b;

                    delta_W_prev{l} = delta_W; delta_b_prev{l} = delta_b;
                end

                % Guardar Históricos y Mostrar Progreso (menos frecuente)
                errorCurrentEpoch = gather(errorTotal); % Recoger para historial/plot
                historial_error(epoca) = errorCurrentEpoch;

                if mod(epoca, 20) == 0 || epoca == 1 || errorCurrentEpoch <= precision || epoca == maxEpocas
                    if ~isempty(hl) && isvalid(hl)
                        addpoints(hl, epoca, errorCurrentEpoch); drawnow('limitrate');
                    end
                    % Guardar historial de pesos (opcional, consume memoria)
                    % if useGPU; historial_pesos{end+1} = cellfun(@gather, obj.Pesos, 'UniformOutput', false); else; historial_pesos{end+1} = obj.Pesos; end
                    % if useGPU; historial_sesgos{end+1} = cellfun(@gather, obj.Sesgos, 'UniformOutput', false); else; historial_sesgos{end+1} = obj.Sesgos; end
                    % savedEpochs(end+1) = epoca;
                end

                if mod(epoca, 100) == 0 || errorCurrentEpoch <= precision || epoca == 1
                    fprintf('Época: %d, Error MSE/2: %.8g\n', epoca, errorCurrentEpoch);
                end
            end % Fin while

            % Finalización
            if epoca == maxEpocas; warning('Entrenamiento detenido: Límite máximo épocas (%d).', maxEpocas); end
            errorFinal = gather(errorTotal); % Recoger final

            % Recoger Pesos/Sesgos de GPU si es necesario
            if useGPU
                for l = 1:(obj.NumCapas - 1); obj.Pesos{l} = gather(obj.Pesos{l}); obj.Sesgos{l} = gather(obj.Sesgos{l}); end
                fprintf('Pesos/Sesgos finales recogidos de GPU.\n');
            end

            fprintf('\n--- Entrenamiento Finalizado ---\n');
            fprintf('Épocas totales: %d\n', epoca);
            fprintf('Error final MSE/2: %.8g\n', errorFinal);

            % Ajustar plot final
             if ~isempty(hl) && isvalid(hl)
                 if epoca > 1; axesHandleError.XLim = [1 epoca]; else; axesHandleError.XLim = [0.5 1.5]; end
                 axesHandleError.YLimMode = 'auto'; drawnow; 
             end
        end % Fin entrenar

        % Guardar pesos y arquitectura en archivo de texto
        function guardarPesos(obj, filepath)
            % Asegurarse que los pesos/sesgos estén en CPU antes de guardar
             pesos_cpu = obj.Pesos; sesgos_cpu = obj.Sesgos;
             if isa(pesos_cpu{1},'gpuArray'); pesos_cpu = cellfun(@gather, pesos_cpu, 'UniformOutput', false); end
             if isa(sesgos_cpu{1},'gpuArray'); sesgos_cpu = cellfun(@gather, sesgos_cpu, 'UniformOutput', false); end

             if isempty(pesos_cpu) || isempty(sesgos_cpu); warning('Pesos/Sesgos vacíos, nada que guardar.'); return; end

             try
                 fid = fopen(filepath, 'wt'); if fid == -1; error('No se pudo abrir "%s" para guardar.', filepath); end
                 fprintf(fid, '%d ', obj.Arquitectura); fprintf(fid, '\n'); % Arquitectura
                 for l = 1:(obj.NumCapas - 1) % Pesos y Sesgos
                     W = pesos_cpu{l}; b = sesgos_cpu{l};
                     fprintf(fid, '%.15g\n', W(:)); % Escribir W por columnas
                     fprintf(fid, '%.15g\n', b(:)); % Escribir b
                 end
                 fclose(fid);
                 fprintf('Pesos guardados correctamente en "%s".\n', filepath);
             catch ME
                 warning('Error guardando pesos en %s: %s', filepath, ME.message);
                 if exist('fid','var') && fid ~= -1; try fclose(fid); catch; end; end
             end
        end

        % Cargar pesos y arquitectura desde archivo
        function [obj, arquitectura] = cargarPesos(obj, filepath)
             if ~isfile(filepath); error('Archivo pesos "%s" no encontrado.', filepath); end
             fid = fopen(filepath, 'r'); if fid == -1; error('No se pudo abrir archivo pesos.'); end

             try
                 header = fgetl(fid); arquitectura = str2num(header); %#ok<ST2NM>
                 if isempty(arquitectura) || ~isvector(arquitectura) || any(arquitectura<=0); error('No se pudo leer arquitectura válida de "%s".', filepath); end
                 data = fscanf(fid, '%f'); fclose(fid);
                 if isempty(data); error('No se encontraron datos numéricos en "%s".', filepath); end

                 obj.Arquitectura = arquitectura; obj.NumCapas = numel(arquitectura);
                 obj.Pesos = cell(1, obj.NumCapas - 1); obj.Sesgos = cell(1, obj.NumCapas - 1);
                 idx = 1;
                 for l = 1:(obj.NumCapas-1)
                     nOut = arquitectura(l+1); nIn = arquitectura(l);
                     totalW = nOut * nIn; totalB = nOut;
                     if idx + totalW - 1 > numel(data); error('Datos insuficientes para pesos W%d.', l); end
                     obj.Pesos{l} = reshape(data(idx : idx + totalW - 1), [nOut, nIn]); % Reshape asume orden columna
                     idx = idx + totalW;
                     if idx + totalB - 1 > numel(data); error('Datos insuficientes para sesgos b%d.', l); end
                     obj.Sesgos{l} = reshape(data(idx : idx + totalB - 1), [nOut, 1]);
                     idx = idx + totalB;
                 end
                 if idx - 1 ~= numel(data); warning('Datos sobrantes en archivo pesos "%s".', filepath); end
                 fprintf('Pesos y sesgos cargados desde "%s".\n', filepath);
             catch ME
                 if exist('fid','var') && fid ~= -1; try fclose(fid); catch; end; end
                 obj.Pesos = {}; obj.Sesgos = {}; % Limpiar en error
                 fprintf(2,'Error al cargar pesos desde %s: %s\n', filepath, ME.message);
                 rethrow(ME);
             end
             % Nota: Las funciones de activación NO se cargan aquí, deben asignarse después si es necesario.
        end

        % Evaluar clasificación (Matriz de Confusión)
        function [ConfMatPercent, ConfMatCounts] = evaluarClasificacion(obj, X_eval, Y_eval)
            % Asegurar que los datos estén en CPU para evaluación estándar
            if isa(X_eval,'gpuArray'); X_eval = gather(X_eval); end
            if isa(Y_eval,'gpuArray'); Y_eval = gather(Y_eval); end
            pesos_eval = obj.Pesos; % Usar pesos actuales del objeto
            if isa(pesos_eval{1},'gpuArray'); pesos_eval = cellfun(@gather, pesos_eval, 'UniformOutput', false); end % Recoger si están en GPU
            sesgos_eval = obj.Sesgos;
             if isa(sesgos_eval{1},'gpuArray'); sesgos_eval = cellfun(@gather, sesgos_eval, 'UniformOutput', false); end

            % Crear una copia temporal en CPU para evaluar (si es necesario)
            temp_obj_cpu = obj; % Copia superficial
            temp_obj_cpu.Pesos = pesos_eval; % Asignar pesos CPU
            temp_obj_cpu.Sesgos = sesgos_eval; % Asignar sesgos CPU
            % Las funciones de activación deberían ser handles válidos en CPU

            if isempty(temp_obj_cpu.Pesos) || isempty(temp_obj_cpu.Sesgos); error('Red no inicializada/entrenada para evaluar.'); end
            if size(X_eval, 1) ~= temp_obj_cpu.Arquitectura(1) || size(Y_eval, 1) ~= temp_obj_cpu.Arquitectura(end); error('Dimensiones X_eval/Y_eval no coinciden con arquitectura.'); end
            if size(X_eval, 2) ~= size(Y_eval, 2); error('Número de patrones entrada/salida evaluación no coincide.'); end

            numPatronesEval = size(X_eval, 2);
            ConfMatCounts = zeros(numPatronesEval, numPatronesEval); % Asume una clase por patrón de salida único
            Y_unique_targets = unique(Y_eval', 'rows')'; % Encontrar targets únicos
            numClasses = size(Y_unique_targets, 2);
             if numClasses ~= numPatronesEval
                 warning('Evaluación: El número de patrones de evaluación (%d) no coincide con el número de targets únicos (%d). La matriz de confusión puede ser engañosa.', numPatronesEval, numClasses);
                 % Podríamos ajustar la matriz ConfMatCounts a [numClasses x numClasses] pero requiere mapeo
             end

            fprintf('Evaluando clasificación (%d patrones)...\n', numPatronesEval);
            for i = 1:numPatronesEval
                x_i = X_eval(:, i);
                Yo_i = temp_obj_cpu.feedforward(x_i); % Feedforward en CPU con la copia temporal
                distancias = sum((Yo_i - Y_eval).^2, 1); % Distancia a TODOS los targets originales
                [~, k_pred_idx] = min(distancias); % Índice del target MÁS CERCANO
                % Asignar al índice i (asumiendo patrón i es clase i) vs índice k_pred_idx
                ConfMatCounts(i, k_pred_idx) = ConfMatCounts(i, k_pred_idx) + 1;
                if mod(i, 100) == 0; fprintf('.'); end % Progreso
            end
            fprintf(' Evaluación completada.\n');

            % Calcular porcentajes
            sumRows = sum(ConfMatCounts, 2); nonZeroRows = sumRows > 0;
            ConfMatPercent = zeros(size(ConfMatCounts));
            ConfMatPercent(nonZeroRows, :) = (ConfMatCounts(nonZeroRows, :) ./ sumRows(nonZeroRows)) * 100;
            ConfMatPercent(~nonZeroRows, :) = NaN; % Marcar filas no evaluadas
            if any(~nonZeroRows); warning('Algunos patrones de entrada no tuvieron instancias en la matriz de confusión.'); end
        end
    end % Fin methods

    methods (Static)
        % Leer configuración desde archivo
        function config = leerConfig(filepath)
             config = struct(); if ~isfile(filepath); warning('Archivo config "%s" no encontrado.', filepath); return; end
             try; lines = readlines(filepath, "Encoding", "UTF-8"); catch ME; warning('No se pudo leer config "%s": %s.', filepath, ME.message); return; end
             fprintf('Leyendo configuración desde "%s"...\n', filepath); keysFound = {};
             for i = 1:length(lines); line = strtrim(lines{i}); if isempty(line) || startsWith(line, '#'); continue; end; parts = split(line, '='); if length(parts) < 2; warning('Línea mal formada: "%s". Ignorando.', line); continue; end; key = lower(strtrim(parts{1})); valueStr = strtrim(strjoin(parts(2:end), '='));
                 try; switch key; case 'arquitectura'; val = str2num(valueStr); if ~isempty(val) && isnumeric(val) && isvector(val) && all(val > 0) && all(fix(val)==val); config.arquitectura = val; keysFound{end+1} = key; else; warning('Valor inválido para arquitectura: "%s".', valueStr); end; case 'alpha'; val = str2double(valueStr); if ~isnan(val) && val > 0; config.alpha = val; keysFound{end+1} = key; else; warning('Valor inválido para alpha: "%s".', valueStr); end; case 'precision'; val = str2double(valueStr); if ~isnan(val) && val > 0; config.precision = val; keysFound{end+1} = key; else; warning('Valor inválido para precision: "%s".', valueStr); end; case 'beta'; val = str2double(valueStr); if ~isnan(val) && val >= 0 && val < 1; config.beta = val; keysFound{end+1} = key; else; warning('Valor inválido para beta: "%s".', valueStr); end; otherwise; warning('Clave desconocida "%s". Ignorando.', key); end; catch ME; warning('Error procesando clave "%s": %s.', key, ME.message); end; end
             if ~isempty(fieldnames(config)); fprintf('Configuración cargada (%s).\n', strjoin(keysFound, ', ')); else; warning('No se cargó configuración válida desde "%s".', filepath); end
        end

        % Leer funciones (strings) desde archivo
        function funciones = leerFunciones(filepath)
             funciones = {}; fid = fopen(filepath, 'r'); if fid == -1; error('No se pudo abrir archivo funciones: %s', filepath); end
             try; while ~feof(fid); line = strtrim(fgetl(fid)); if ~isempty(line) && ~startsWith(line, '#'); funciones{end+1, 1} = line; end; end; catch ME; fclose(fid); error('Error leyendo funciones: %s', ME.message); end; fclose(fid);
             fprintf('Funciones cargadas desde ''%s'': %d funciones.\n', filepath, length(funciones));
        end

        % Guardar configuración en archivo
        function guardarConfig(filepath, config_struct)
             fprintf('Guardando configuración en "%s"...', filepath);
             fid = fopen(filepath, 'wt'); if fid == -1; warning('No se pudo abrir "%s" para escribir config.', filepath); return; end
             try
                 fprintf(fid, '# Última config. entrenamiento (%s)\n', datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
                 if isfield(config_struct, 'alpha'); fprintf(fid, 'alpha = %.5g\n', config_struct.alpha); end
                 if isfield(config_struct, 'beta'); fprintf(fid, 'beta = %.5g\n', config_struct.beta); end
                 if isfield(config_struct, 'precision'); fprintf(fid, 'precision = %.5g\n', config_struct.precision); end
                 if isfield(config_struct, 'arquitectura')
                     arch_str = sprintf('%d ', config_struct.arquitectura); fprintf(fid, 'arquitectura = %s\n', strtrim(arch_str)); % Guardar sin corchetes
                 end
                 fclose(fid); fprintf(' Configuración guardada.\n');
             catch ME
                  warning('Error guardando config: %s', ME.message);
                  if exist('fid','var') && fid ~= -1; try fclose(fid); catch; end; end
             end
        end
    end % Fin methods (Static)

end % Fin classdef