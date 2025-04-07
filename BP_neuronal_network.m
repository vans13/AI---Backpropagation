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
        
        function [salidaFinal, activaciones, nets] = feedforward(obj, entrada)
            % Realiza la pasada hacia adelante (feedforward)
            % 'entrada' debe ser un vector columna
            if ~iscolumn(entrada)
                entrada = entrada'; % Asegurar que la entrada sea vector columna
            end
            if size(entrada, 1) ~= obj.Arquitectura(1)
                error('Dimensiones de entrada no coinciden con la arquitectura.');
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

        function [obj, epoca, errorFinal, historial_error, historial_pesos, historial_sesgos] = entrenar(obj, X_train, Y_train, alpha, precision)
                        % Entrena la red usando backpropagation
            % DEVUELVE: 
            %   obj: El objeto red entrenado
            %   epoca: El número final de épocas realizadas
            %   errorFinal: El error MSE final alcanzado
            %   historial_error: Vector con el MSE de cada época
            %   historial_pesos: Cell array con los pesos de cada época
            %   historial_sesgos: Cell array con los sesgos de cada época
            % X_train: matriz de entradas (patrones por fila)
            % Y_train: matriz de salidas deseadas (patrones por fila)
            numPatrones = size(X_train, 1);
            if size(Y_train, 1) ~=numPatrones
                 error('Número de patrones de entrada y salida no coincide.');
            end
            if size(X_train, 2) ~= obj.Arquitectura(1)
                error('Número de características de entrada no coincide con arquitectura.');
            end
            if size(Y_train, 2) ~= obj.Arquitectura(end)
                error('Número de características de salida no coincide con arquitectura.');
            end
            epoca = 0;  % Inicializar las épocas
            errorTotal = inf;  % Inicializar error alto
            % --- Inicializar Históricos ---
            historial_error = []; 
            historial_pesos = {}; % Cell array para guardar cell arrays de pesos
            historial_sesgos = {}; % Cell array para guardar cell arrays de sesgos

            fprintf('\n--- Iniciando Entrenamiento ---\n');
            fprintf('Patrones: %d, Alpha: %.4f, Precisión requerida: %.g\n', numPatrones, alpha, precision);
            
            while errorTotal > precision
                epoca = epoca + 1;
                errorActual = 0;
                for i = 1:numPatrones
                    x = X_train(i,:);
                    Yd = Y_train(i,:);
                    
                    %Primero se realiza el feedforward
                    [Yo, activaciones, nets] = obj.feedforward(x);

                    %Cálculo del error para el patrón
                    errorPatron = Yd - Yo;
                    errorActual = errorActual + sum(errorPatron.^2);
                    
                    %Backpropagation - Cálculo de deltas
                    deltas = cell(1,obj.NumCapas - 1);
                    derivadaSalida = obj.DerivadasActivacion{end}(activaciones{end});
                    deltas{end} = errorPatron .* derivadaSalida;
                    for l = (obj.NumCapas - 2):-1:1
                        W_siguiente = obj.Pesos{l+1};
                        delta_siguiente = deltas{l+1};
                        activacion_l = activaciones{l+1};
                        derivadaOculta = obj.DerivadasActivacion{l}(activacion_l);
                        errorPropagado = W_siguiente' * delta_siguiente;
                        deltas{l} = errorPropagado .* derivadaOculta;
                    end
                    for l = 1:(obj.NumCapas - 1) % Para cada capa l (desde la 1ra oculta hasta la salida)
                        activacion_previa = activaciones{l}; % Activación de la capa anterior (a^{l-1}), vector columna
                        delta_l = deltas{l};                 % Delta calculado para la capa l actual, vector columna

                        % Calcular gradientes
                        grad_W = delta_l * activacion_previa'; % Gradiente pesos W^l: (n_l x 1) * (1 x n_{l-1}) = (n_l x n_{l-1})
                        grad_b = delta_l;                      % Gradiente bias b^l: (n_l x 1)

                        % Actualizar pesos y sesgos de la capa l
                        obj.Pesos{l} = obj.Pesos{l} + alpha * grad_W;
                        obj.Sesgos{l} = obj.Sesgos{l} + alpha * grad_b; 
                    end
                end
                % Calcular error total promedio de la época
                errorTotal = errorActual / numPatrones;
                % --- Guardar en Históricos (al final de la época) ---
                historial_error(epoca) = errorTotal;
                historial_pesos{epoca} = obj.Pesos;   % Guarda el cell array actual de pesos
                historial_sesgos{epoca} = obj.Sesgos; % Guarda el cell array actual de sesgos
                % Imprimir progreso
                if mod(epoca, 100) == 0 || errorTotal <= precision
                  fprintf('Época: %d, Error MSE: %.8f\n', epoca, errorTotal);
                end
                % Condición de seguridad
                if epoca > 5000000 
                  warning('Entrenamiento detenido: Límite máximo de épocas alcanzado.');
                  break;
                end
            end
            errorFinal = errorTotal; % Guardar el último error calculado
            fprintf('\n--- Entrenamiento Finalizado ---\n');
            fprintf('Épocas totales: %d\n', epoca);
            fprintf('Error final MSE: %.8f\n', errorFinal);
        end

    end

    methods (Static)
        function config = leerConfig(filepath)
            % Inicializar estructura de configuración
            config = struct();
            if ~isfile(filepath)
                 warning('Archivo de configuración "%s" no encontrado.', filepath);
                 return;
            end
            %Leer todas las líneas del archivo
            try
                lines = readlines(filepath, "Encoding", "UTF-8"); 
            catch ME
                error('No se pudo leer el archivo de configuración "%s": %s', filepath, ME.message);
            end
            %Procesar cada línea
            for i = 1:length(lines)
                line = strtrim(lines{i}); % readlines devuelve cell array o string array
                %Ignorar líneas vacías y comentarios (usando '#')
                if isempty(line) || startsWith(line, '#') 
                    continue;
                end
                % Dividir línea en clave y valor por el PRIMER '='
                key_value = strsplit(lines{i}, '=');

                if length(key_value) < 2
                    warning('Línea mal formada (sin "="): "%s" en "%s". Ignorando.', line, filepath); 
                    continue; % Saltar línea mal formada
                end
                key = strtrim(key_value{1});
                value = strtrim(key_value{2}); % Valor inicial como string
                % Validar que la clave sea un nombre de variable válido
                if ~isvarname(key)
                     warning('Clave inválida "%s" en "%s". Ignorando.', key, filepath);
                     continue;
                end
                % 8. Convertir los valores según la clave usando switch
                try % Envuelve la conversión en try-catch para robustez
                    switch key
                        case 'arquitectura' 
                            converted_value = str2num(value); 
                            % Validar conversión de arquitectura
                            if isempty(converted_value) || ~isnumeric(converted_value) || ~isrow(converted_value)
                                error('Valor inválido o no es vector fila para arquitectura: "%s"', value);
                            end
                            config.arquitectura = converted_value;

                        case 'alpha' 
                            converted_value = str2double(value); 
                             % Validar conversión de alfa
                            if isnan(converted_value)
                                 error('Valor numérico inválido para alfa: "%s"', value);
                            end
                            config.alpha = converted_value;

                        case 'precision' 
                             converted_value = str2double(value);
                             % Validar conversión de precisión
                             if isnan(converted_value) || converted_value <= 0
                                 error('Valor inválido para la precisión (debe ser positivo): "%s"', value);
                             end
                             config.precision = converted_value; % Guardar como número
                             
                        otherwise
                            % Ignorar con advertencia (actual)
                             warning('Clave desconocida "%s" en "%s". Ignorando.', key, filepath);
                    end
                catch ME
                    % Capturar errores durante la conversión o validación
                    warning('Error procesando clave "%s" con valor "%s" en "%s": %s. Ignorando entrada.', key, value, filepath, ME.message);
                end
            end
            if ~isempty(fieldnames(config))
                fprintf('Configuración cargada desde "%s":\n', filepath);
                disp(config);
            else
                 warning('No se cargó ninguna configuración válida desde "%s".', filepath);
            end
        end

        function funciones = leerFunciones(filepath)
            % Lee las funciones (una por línea) y devuelve un cell array de strings.
            funciones = {};
            fid = fopen(filepath, 'r');
            if fid == -1
                error('No se pudo abrir el archivo de funciones: %s', filepath);
            end
            while ~feof(fid)
                line = strtrim(fgetl(fid));
                 if ~isempty(line) && ~startsWith(line, '#')
                     funciones{end+1, 1} = line;
                 end
            end
            fclose(fid);
             fprintf('Funciones cargadas desde ''%s'': %d funciones.\n', filepath, length(funciones));
        end
    end
end