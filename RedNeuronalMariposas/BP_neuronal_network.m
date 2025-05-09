% Realizado por Juan Esteban Fuentes, Laura Latorre y Duvan Santiago Matallana
% Clase optimizada de backpropagation sin Deep Learning Toolbox,
% con vectorización, paralelismo y sesgos negativos, manteniendo todas las interfaces.
classdef BP_neuronal_network
    properties
        Arquitectura          % Vector [in, h1, ..., hn, out]
        NumCapas              % Número total de capas
        Pesos                 % Cell array {W1, W2, ...}
        Sesgos                % Cell array {b1, b2, ...} (siempre <= 0)
        FuncionesActivacion   % Cell array {@(net)..., ...}
        DerivadasActivacion   % Cell array {@(act)..., ...}
        UseGPU logical = false % Flag para usar GPU si está disponible
    end

    methods
        %% Constructor
        function obj = BP_neuronal_network(arquitectura, funcActStr, derivActStr)
            if nargin > 0
                obj.Arquitectura = arquitectura;
                obj.NumCapas = numel(arquitectura);
                if numel(funcActStr) ~= obj.NumCapas-1 || numel(derivActStr) ~= obj.NumCapas-1
                    error('Número de funciones/derivadas debe coincidir con capas ocultas + salida.');
                end
                rng('shuffle');
                % Inicialización pesos y sesgos
                obj.Pesos = cell(1, obj.NumCapas-1);
                obj.Sesgos = cell(1, obj.NumCapas-1);
                for l = 1:obj.NumCapas-1
                    nIn = arquitectura(l);
                    nOut = arquitectura(l+1);
                    obj.Pesos{l} = randn(nOut, nIn)*0.1;
                    obj.Sesgos{l} = -abs(randn(nOut,1)*0.1);
                    obj.FuncionesActivacion{l} = str2func(['@(net) ' funcActStr{l}]);
                    obj.DerivadasActivacion{l} = str2func(['@(act) ' derivActStr{l}]);
                end
                disp('Red inicializada con vectorización y sesgos negativos.');
            end
        end

        %% Establecer pesos y sesgos iniciales
        function obj = setInitialWeights(obj, initialPesos, initialSesgos)
            if isempty(obj.Arquitectura)
                error('Arquitectura no definida para setInitialWeights.');
            end
            if numel(initialPesos)~=obj.NumCapas-1 || numel(initialSesgos)~=obj.NumCapas-1
                error('Número de cell arrays de pesos/sesgos iniciales no coincide.');
            end
            for l=1:obj.NumCapas-1
                expW = [obj.Arquitectura(l+1), obj.Arquitectura(l)];
                expB = [obj.Arquitectura(l+1), 1];
                if ~isequal(size(initialPesos{l}), expW)
                    error('Tamaño pesos iniciales capa %d incorrecto.', l);
                end
                if ~isequal(size(initialSesgos{l}), expB)
                    error('Tamaño sesgos iniciales capa %d incorrecto.', l);
                end
                obj.Pesos{l}  = initialPesos{l};
                obj.Sesgos{l} = min(initialSesgos{l}, 0);
            end
            disp('Pesos y sesgos iniciales establecidos externamente.');
        end

        %% Feedforward Vectorizado
        function [salidaFinal, activaciones, nets] = feedforward(obj, X)
            if size(X,1)~=obj.Arquitectura(1)
                error('Feedforward: dimensiones de entrada no coinciden con arquitectura.');
            end
            activaciones = cell(1, obj.NumCapas);
            nets = cell(1, obj.NumCapas-1);
            a = X;
            activaciones{1} = a;
            for l=1:obj.NumCapas-1
                net = obj.Pesos{l}*a + obj.Sesgos{l};
                a = obj.FuncionesActivacion{l}(net);
                nets{l} = net;
                activaciones{l+1} = a;
            end
            salidaFinal = a;
        end

        %% Entrenamiento con validaciones, plotting y paralelismo
        function [obj, epoca_final, error_final_mse, historial_error_mse, historial_norma_pesos, epocas_para_norma_pesos, savedEpochs_no_usado] = entrenar_con_gui_update(obj, X_train, Y_train, alpha_lr, precision_mse_deseada, beta_momentum, axesHandleErrorPlot, max_epocas_usuario, app_handle_gui)
            % Validaciones de entrada
            narginchk(8,9);
            if isempty(beta_momentum), beta_momentum = 0; end

            % --- Preparación para el entrenamiento ---
            use_gui_plot = isgraphics(axesHandleErrorPlot);
            if use_gui_plot
                cla(axesHandleErrorPlot);
                error_line_plot = animatedline(axesHandleErrorPlot, 'Color', [0 0.4470 0.7410], 'Marker', '.', 'LineStyle', '-'); % Color azul MATLAB
                title(axesHandleErrorPlot, 'Error (MSE/2) vs. Épocas');
                xlabel(axesHandleErrorPlot, 'Época');
                ylabel(axesHandleErrorPlot, 'MSE/2');
                grid(axesHandleErrorPlot, 'on');
                axesHandleErrorPlot.XAxis.Visible = 'on'; % Forzar visibilidad
                axesHandleErrorPlot.YAxis.Visible = 'on'; % Forzar visibilidad
                axis(axesHandleErrorPlot, 'tight');
            end

            % --- Habilitación y Verificación de GPU ---
            useGPU = false; % Inicializar como false
            if obj.UseGPU % Solo intentar si la propiedad del objeto está activa
                try
                    if gpuDeviceCount > 0
                        gpu_dev = gpuDevice; % Obtener info del dispositivo
                        fprintf('INFO: GPU detectada (%s) y habilitada para entrenamiento.\n', gpu_dev.Name);
                        useGPU = true;
                    else
                        warning('BP_NN:GPU', 'GPU solicitada (UseGPU=true) pero no se detectaron dispositivos GPU compatibles. Usando CPU.');
                        obj.UseGPU = false; % Desactivar para el resto de esta ejecución
                    end
                catch ME_gpu
                    warning('BP_NN:GPU', 'Error al verificar/usar la GPU: %s. Usando CPU.', ME_gpu.message);
                    obj.UseGPU = false; % Desactivar
                end
            end
            % --- FIN Habilitación GPU ---

            % Transferir datos a GPU si está habilitada
            original_X_class = class(X_train); % Guardar clase original por si acaso
            original_Y_class = class(Y_train);
            if useGPU
                try
                    X_train = gpuArray(X_train);
                    Y_train = gpuArray(Y_train);
                    % Transferir pesos, sesgos y deltas previos a GPU
                    obj.Pesos  = cellfun(@gpuArray, obj.Pesos, 'UniformOutput', false);
                    obj.Sesgos = cellfun(@gpuArray, obj.Sesgos, 'UniformOutput', false);
                    delta_pesos_previos = cellfun(@gpuArray, cellfun(@(w) zeros(size(w)), obj.Pesos, 'UniformOutput', false), 'UniformOutput', false);
                    delta_sesgos_previos = cellfun(@gpuArray, cellfun(@(b) zeros(size(b)), obj.Sesgos, 'UniformOutput', false), 'UniformOutput', false);
                    fprintf('INFO: Datos, pesos y estados transferidos a la GPU.\n');
                catch ME_gpu_transfer
                    warning('BP_NN:GPU', 'Error al transferir datos/pesos a la GPU: %s. Reintentando en CPU.', ME_gpu_transfer.message);
                    useGPU = false; % Fallback a CPU
                    obj.UseGPU = false;
                    % Asegurarse que los pesos/sesgos estén en CPU
                    obj.Pesos = cellfun(@gatherIfNeeded, obj.Pesos, 'UniformOutput', false);
                    obj.Sesgos = cellfun(@gatherIfNeeded, obj.Sesgos, 'UniformOutput', false);
                    % Inicializar deltas en CPU
                    delta_pesos_previos = cellfun(@(w) zeros(size(w), original_X_class), obj.Pesos, 'UniformOutput', false); % Usar clase original
                    delta_sesgos_previos = cellfun(@(b) zeros(size(b), original_X_class), obj.Sesgos, 'UniformOutput', false);
                end
            else
                % Inicializar deltas en CPU si no se usa GPU
                delta_pesos_previos = cellfun(@(w) zeros(size(w), original_X_class), obj.Pesos, 'UniformOutput', false);
                delta_sesgos_previos = cellfun(@(b) zeros(size(b), original_X_class), obj.Sesgos, 'UniformOutput', false);
            end

            num_capas_entrenamiento = obj.NumCapas - 1;
            num_muestras = size(X_train, 2);

            % Arrays de historial siempre en CPU
            historial_error_mse = zeros(1, max_epocas_usuario);
            historial_norma_pesos = zeros(1, max_epocas_usuario);
            epocas_para_norma_pesos = zeros(1, max_epocas_usuario);

            epoca_actual = 0;
            error_actual_mse = inf;
            savedEpochs_no_usado = [];

            fprintf('--- Iniciando Entrenamiento (Device: %s, max_epochs: %d, target_mse: %.6g, alpha: %.4g, beta: %.3g) ---\n', ...
                iif(useGPU, 'GPU', 'CPU'), max_epocas_usuario, precision_mse_deseada, alpha_lr, beta_momentum); % Usando iif helper

            % --- Bucle Principal de Entrenamiento ---
            while error_actual_mse > precision_mse_deseada && epoca_actual < max_epocas_usuario
                epoca_actual = epoca_actual + 1;

                % Verificar señal de parada desde la GUI
                should_stop_training = false;
                if nargin == 9 && isobject(app_handle_gui) && isvalid(app_handle_gui) && ismethod(app_handle_gui, 'getStopTrainingStatus')
                    should_stop_training = app_handle_gui.getStopTrainingStatus();
                end
                if should_stop_training
                    fprintf('INFO: Entrenamiento detenido por el usuario en época %d.\n', epoca_actual);
                    break;
                end

                % Feedforward (opera en CPU o GPU dependiendo de los datos/pesos)
                [salida_predicha_Y, activaciones_capas, ~] = obj.feedforward(X_train);
                error_epoca = Y_train - salida_predicha_Y;

                % Backpropagation (opera en CPU o GPU)
                deltas_error_capas = cell(1, num_capas_entrenamiento);
                deltas_error_capas{num_capas_entrenamiento} = error_epoca .* obj.DerivadasActivacion{num_capas_entrenamiento}(activaciones_capas{end});
                for k = (num_capas_entrenamiento - 1):-1:1
                    deltas_error_capas{k} = (obj.Pesos{k+1}' * deltas_error_capas{k+1}) .* obj.DerivadasActivacion{k}(activaciones_capas{k+1});
                end

                % Actualización de Pesos y Sesgos (opera en CPU o GPU)
                norma_total_pesos_epoca_sq = iif(useGPU, gpuArray(0), 0); % Inicializar en el dispositivo correcto
                for k = 1:num_capas_entrenamiento
                    gradiente_pesos = (deltas_error_capas{k} * activaciones_capas{k}') / num_muestras;
                    gradiente_sesgos = sum(deltas_error_capas{k}, 2) / num_muestras;

                    dw = alpha_lr * gradiente_pesos + beta_momentum * delta_pesos_previos{k};
                    db = alpha_lr * gradiente_sesgos + beta_momentum * delta_sesgos_previos{k};

                    obj.Pesos{k} = obj.Pesos{k} + dw;
                    obj.Sesgos{k} = min(0, obj.Sesgos{k} + db);

                    delta_pesos_previos{k} = dw;
                    delta_sesgos_previos{k} = db;

                    % Calcular norma (asegurarse que funciona en GPU)
                    norma_total_pesos_epoca_sq = norma_total_pesos_epoca_sq + sum(obj.Pesos{k}(:).^2);
                end

                % Calcular error y norma (recoger de GPU si es necesario)
                error_actual_mse = gather(0.5 * sum(error_epoca(:).^2) / num_muestras); % Gather para asegurar valor CPU
                historial_error_mse(epoca_actual) = error_actual_mse; % Guardar en array CPU

                norma_actual_pesos = gather(sqrt(norma_total_pesos_epoca_sq)); % Gather para asegurar valor CPU
                historial_norma_pesos(epoca_actual) = norma_actual_pesos; % Guardar en array CPU
                epocas_para_norma_pesos(epoca_actual) = epoca_actual;

                % Actualizar GUI y consola (cada N épocas o en puntos clave)
                if mod(epoca_actual, 200) == 0 || epoca_actual == 1 || error_actual_mse <= precision_mse_deseada || epoca_actual == max_epocas_usuario
                    if use_gui_plot
                        addpoints(error_line_plot, epoca_actual, error_actual_mse);
                        % Forzar un re-ajuste de los límites cada cierto tiempo para animatedline
                        if mod(epoca_actual, 25) == 0 || epoca_actual == 1 || error_actual_mse <= precision_mse_deseada || epoca_actual == max_epocas_usuario
                            if use_gui_plot % Actualización de UIAxes2 (error plot)
                                addpoints(error_line_plot, epoca_actual, error_actual_mse);
                                if mod(epoca_actual, 50) == 0 || epoca_actual == 1 % Ajustar ejes de la gráfica de error con menos frecuencia que addpoints
                                    axis(axesHandleErrorPlot,'auto'); % 'tight' puede ser muy variable, 'auto' es más estable
                                    drawnow('limitrate'); % Actualizar gráfica de error aquí mismo
                                end
                            end
                            % Llamar a la función de actualización de la GUI de la app
                            if nargin == 9 && isvalid(app_handle_gui) && ismethod(app_handle_gui, 'update_live_training_metrics')
                                app_handle_gui.update_live_training_metrics(epoca_actual, error_actual_mse, norma_actual_pesos);
                            end
                            fprintf('Ep: %d/%d, MSE/2: %.7g, Norma Pesos: %.4g\n', epoca_actual, max_epocas_usuario, error_actual_mse, norma_actual_pesos);
                        end
                        drawnow('limitrate');
                    end
                    % Llamar a la función de actualización de la GUI si existe
                    if nargin == 9 && isvalid(app_handle_gui) && ismethod(app_handle_gui, 'update_live_training_metrics')
                        app_handle_gui.update_live_training_metrics(epoca_actual, error_actual_mse, norma_actual_pesos); % <<--- ESTA ES LA LLAMADA
                    end
                    fprintf('Ep: %d/%d, MSE/2: %.7g, Norma Pesos: %.4g\n', epoca_actual, max_epocas_usuario, error_actual_mse, norma_actual_pesos);
                end
            end % Fin del bucle while

            % --- Finalización del Entrenamiento ---
            epoca_final = epoca_actual;
            error_final_mse = error_actual_mse; % Ya está en CPU

            % Recortar historiales (ya están en CPU)
            historial_error_mse = historial_error_mse(1:epoca_final);
            historial_norma_pesos = historial_norma_pesos(1:epoca_final);
            epocas_para_norma_pesos = epocas_para_norma_pesos(1:epoca_final);

            % --- Recoger Pesos y Sesgos de la GPU a la CPU ---
            if useGPU % Si se estaba usando la GPU, traer los pesos finales a la CPU
                try
                    obj.Pesos = cellfun(@gather, obj.Pesos, 'UniformOutput', false);
                    obj.Sesgos = cellfun(@gather, obj.Sesgos, 'UniformOutput', false);
                    fprintf('INFO: Pesos y sesgos finales recogidos de la GPU.\n');
                catch ME_gather_final
                    warning('BP_NN:GPU', 'Error al recoger pesos/sesgos finales de la GPU: %s', ME_gather_final.message);
                end
            end
            obj.UseGPU = false; % Resetear flag interno después del entrenamiento (opcional)

            fprintf('--- Entrenamiento Finalizado ---\nÉpocas completadas: %d, Error MSE/2 final: %.7g\n', epoca_final, error_final_mse);
            if use_gui_plot && epoca_final > 0
                % Actualizar la línea con todos los puntos por si animatedline no lo hizo del todo
                clearpoints(error_line_plot);
                addpoints(error_line_plot, 1:epoca_final, historial_error_mse(1:epoca_final));

                min_epoch_plot = 1;
                max_epoch_plot = max(1, epoca_final);
                xlim(axesHandleErrorPlot, [min_epoch_plot - 0.05*max_epoch_plot, max_epoch_plot + 0.05*max_epoch_plot]); % Añadir un pequeño margen

                relevant_errors = historial_error_mse(1:epoca_final);
                if ~isempty(relevant_errors)
                    min_err_val = min(relevant_errors);
                    max_err_val = max(relevant_errors);
                    if min_err_val == max_err_val % Si el error fue constante
                        padding = max(0.1 * abs(min_err_val), 0.0001); % Evitar padding cero
                        ylim(axesHandleErrorPlot, [min_err_val - padding, max_err_val + padding]);
                    else
                        padding = 0.05 * (max_err_val - min_err_val); % 5% de padding
                        if padding == 0, padding = 0.1 * max(abs(min_err_val), abs(max_err_val)); end % Si max y min son iguales
                        if padding == 0, padding = 0.1; end % Último recurso
                        ylim(axesHandleErrorPlot, [min_err_val - padding, max_err_val + padding]);
                    end
                else
                    ylim(axesHandleErrorPlot, 'auto'); % Default si no hay historial
                end
                axesHandleErrorPlot.XAxis.Visible = 'on'; % Re-asegurar
                axesHandleErrorPlot.YAxis.Visible = 'on'; % Re-asegurar
                drawnow;
            elseif use_gui_plot % Caso donde epoca_final es 0
                cla(axesHandleErrorPlot);
                title(axesHandleErrorPlot, 'Error MSE/2 vs. Épocas (Sin entrenamiento)');
                xlabel(axesHandleErrorPlot, 'Época'); ylabel(axesHandleErrorPlot, 'MSE/2');
                grid(axesHandleErrorPlot, 'on');
                axesHandleErrorPlot.XLimMode = 'auto'; % Permitir que se expanda
                axesHandleErrorPlot.YLimMode = 'auto'; % Permitir que se expanda
                axesHandleErrorPlot.XAxis.Visible = 'on'; axesHandleErrorPlot.YAxis.Visible = 'on';
            end
        end

        %% Guardar pesos y arquitectura en archivo
        function guardarPesos(obj, filepath)
            pesos_cpu = obj.Pesos; sesgos_cpu = obj.Sesgos;
            if isa(pesos_cpu{1},'gpuArray')
                pesos_cpu = cellfun(@gather,pesos_cpu,'Uni',false);
                sesgos_cpu= cellfun(@gather,sesgos_cpu,'Uni',false);
            end
            if isempty(pesos_cpu), warning('Nada que guardar.'); return; end
            fid = fopen(filepath,'wt');
            if fid==-1, error('No se pudo abrir %s.',filepath); end
            fprintf(fid,'%d ',obj.Arquitectura); fprintf(fid,'\n');
            for l=1:obj.NumCapas-1
                fprintf(fid,'%.15g\n',pesos_cpu{l}(:));
                fprintf(fid,'%.15g\n',sesgos_cpu{l}(:));
            end
            fclose(fid);
        end

        %% Cargar pesos y arquitectura desde archivo
        function [obj, arquitectura] = cargarPesos(obj, filepath)
            if ~isfile(filepath), error('Archivo no encontrado.'); end
            fid = fopen(filepath,'r');
            header=fgetl(fid); arquitectura=str2num(header); %#ok<ST2NM>
            data=fscanf(fid,'%f'); fclose(fid);
            obj.Arquitectura=arquitectura; obj.NumCapas=numel(arquitectura);
            obj.Pesos=cell(1,obj.NumCapas-1); obj.Sesgos=cell(1,obj.NumCapas-1);
            idx=1;
            for l=1:obj.NumCapas-1
                nOut=arquitectura(l+1); nIn=arquitectura(l);
                totW=nOut*nIn; totB=nOut;
                obj.Pesos{l}=reshape(data(idx:idx+totW-1),[nOut,nIn]); idx=idx+totW;
                obj.Sesgos{l}=min(reshape(data(idx:idx+totB-1),[nOut,1]),0); idx=idx+totB;
            end
        end

        %% Evaluar clasificación
        function [ConfMatPercent, ConfMatCounts] = evaluarClasificacion(obj, X_eval, Y_eval)
            if isa(X_eval,'gpuArray'), X_eval=gather(X_eval); end
            if isa(Y_eval,'gpuArray'), Y_eval=gather(Y_eval); end
            temp=obj; % copia superficial
            temp.Pesos = cellfun(@gather,obj.Pesos,'Uni',false);
            temp.Sesgos= cellfun(@gather,obj.Sesgos,'Uni',false);
            numP=size(X_eval,2);
            ConfMatCounts=zeros(numP,numP);
            for i=1:numP
                x_i=X_eval(:,i);
                y_pred=temp.feedforward(x_i);
                dist=sum((y_pred - Y_eval).^2,1);
                [~,k]=min(dist);
                ConfMatCounts(i,k)=ConfMatCounts(i,k)+1;
            end
            sumRows=sum(ConfMatCounts,2);
            ConfMatPercent=ConfMatCounts;
            for i=1:numP
                if sumRows(i)>0
                    ConfMatPercent(i,:)=ConfMatCounts(i,:)/sumRows(i)*100;
                else
                    ConfMatPercent(i,:)=NaN;
                end
            end
        end
    end

    methods (Static)
        %% Lectura y guardado de configuración y funciones
        function config = leerConfig(filepath)
            config=struct(); if ~isfile(filepath), return; end
            lines=readlines(filepath,'Encoding','UTF-8');
            for ln=lines'
                s=strtrim(ln{1}); if isempty(s)||s(1)=='#', continue; end
                parts=split(s,'='); key=lower(strtrim(parts{1})); val=strtrim(parts(2));
                switch key
                    case 'arquitectura_mlp_ocultas'
                        config.arquitectura_mlp_ocultas=str2num(char(val)); %#ok<ST2NM>
                    case 'alpha'
                        config.alpha=str2double(val);
                    case 'precision'
                        config.precision=str2double(val);
                    case 'beta'
                        config.beta=str2double(val);
                    case 'theta'
                        config.theta=str2double(val);
                    case 'max_epochs'
                        config.max_epochs=str2double(val);
                end
            end
        end

        function funciones = leerFunciones(filepath)
            funciones={}; fid=fopen(filepath,'r'); if fid==-1, return; end
            while ~feof(fid)
                ln=strtrim(fgetl(fid));
                if ~isempty(ln)&&ln(1)~='#', funciones{end+1}=ln; end
            end
            fclose(fid);
        end

        function guardarConfig(filepath, config)
            fid=fopen(filepath,'wt'); if fid==-1, return; end
            fprintf(fid,'# Config guardada %s\n',datestr(now,'yyyy-mm-dd HH:MM:SS'));
            if isfield(config,'alpha'), fprintf(fid,'alpha = %.5g\n',config.alpha); end
            if isfield(config,'beta'), fprintf(fid,'beta = %.5g\n',config.beta); end
            if isfield(config,'precision'), fprintf(fid,'precision = %.5g\n',config.precision); end
            if isfield(config,'arquitectura')
                fprintf(fid,'arquitectura = %s\n',num2str(config.arquitectura));
            end
            fclose(fid);
        end
    end
end

% --- Helper functions (si se necesitan fuera de la clase, o como métodos privados/estáticos) ---
function result = gatherIfNeeded(data)
% Helper para asegurar que gather solo se llame en gpuArrays
if isa(data, 'gpuArray')
    result = gather(data);
else
    result = data;
end
end

function result = iif(condition, trueVal, falseVal)
% Helper inline if
if condition
    result = trueVal;
else
    result = falseVal;
end
end
