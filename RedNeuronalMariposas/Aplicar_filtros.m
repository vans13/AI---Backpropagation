% Aplicar_filtros.m
function imagen_tratada_final = Aplicar_filtros(imagen_original, kernel_seleccionado_nombre, filtro_color_seleccionado_nombre)

    % Convertir a double para procesamiento si es uint8
    if isa(imagen_original, 'uint8')
        imagen_actual_double = im2double(imagen_original);
    else
        imagen_actual_double = imagen_original; % Asumir que ya es double en [0,1] o similar
    end

    % --- Definición de Kernels ---
    kernels = struct();
    % Kernels 3x3 (Nombres deben coincidir con DropDowns)
    kernels.('Enfoque_Sharpen') = [0 -1 0; -1 5 -1; 0 -1 0];
    kernels.('Deteccion_de_bordes_Laplaciano') = [0 1 0; 1 -4 1; 0 1 0];
    kernels.('Deteccion_de_Bordes_Fuerte') = [-1 -1 -1; -1 8 -1; -1 -1 -1]; % Asegúrate que este nombre esté en el dropdown
    kernels.('Desenfoque_Box_Blur') = ones(3)/9; % Normalizado
    kernels.('Repujado_Emboss_1') = [-2 -1 0; -1 1 1; 0 1 2];
    kernels.('Repujado_Emboss_2') = [-1 -1 0; -1 1 1; 0 1 1];
    kernels.('Prewitt_X') = [-1 0 1; -1 0 1; -1 0 1];
    kernels.('Prewitt_Y') = [-1 -1 -1; 0 0 0; 1 1 1];
    kernels.('Sobel_X') = [-1 0 1; -2 0 2; -1 0 1];
    kernels.('Sobel_Y') = [-1 -2 -1; 0 0 0; 1 2 1];
    kernels.('Realce_Laplaciano') = [-1 -1 -1; -1 9 -1; -1 -1 -1];
    kernels.('Identity_3x3') = [0 0 0; 0 1 0; 0 0 0]; % Añadido
    kernels.('GaussianBlur_3x3') = fspecial('gaussian', 3, 0.5); % Añadido (sigma 0.5 ejemplo)

    % Kernels 5x5 (Nombres deben coincidir con DropDowns)
    kernels.('Desenfoque_Gaussiano_5x5') = fspecial('gaussian', 5, 1); % Renombrado para claridad
    kernels.('Desenfoque_Medio_5x5') = ones(5)/25; % Renombrado
    kernels.('Realce_Sharpen_5x5') = [ -1 -1 -1 -1 -1; -1  2  2  2 -1; -1  2  8  2 -1; -1  2  2  2 -1; -1 -1 -1 -1 -1] / 8;
    kernels.('Relieve_Emboss_5x5') = [-2 -1 0 1 2; -2 -1 0 1 2; -2 -1 1 1 2; -2 -1 0 1 2; -2 -1 0 1 2];
    kernels.('Prewitt_X_5x5') = repmat([-2 -1 0 1 2], 5, 1) / 10; % Normalización?
    kernels.('Prewitt_Y_5x5') = repmat([-2; -1; 0; 1; 2], 1, 5) / 10; % Normalización?
    % ...otros kernels 5x5 que tengas...

    % --- Aplicar Kernel Espacial ---
    if ~isempty(kernel_seleccionado_nombre) && isfield(kernels, kernel_seleccionado_nombre)
        kernel_actual = kernels.(kernel_seleccionado_nombre);
        % Aplicar a cada canal si es RGB, o directamente si es gris
        if size(imagen_actual_double, 3) == 3
            img_R = imfilter(imagen_actual_double(:,:,1), kernel_actual, 'replicate');
            img_G = imfilter(imagen_actual_double(:,:,2), kernel_actual, 'replicate');
            img_B = imfilter(imagen_actual_double(:,:,3), kernel_actual, 'replicate');
            imagen_actual_double = cat(3, img_R, img_G, img_B);
        else % Escala de grises
            imagen_actual_double = imfilter(imagen_actual_double, kernel_actual, 'replicate');
        end
        imagen_actual_double = mat2gray(imagen_actual_double); % Normalizar a [0,1] después del filtro
    elseif ~isempty(kernel_seleccionado_nombre)
         warning('Kernel "%s" no encontrado en la lista interna de Aplicar_filtros.', kernel_seleccionado_nombre);
    end

    % --- Aplicar Filtro de Color ---
    % La imagen de entrada a esta sección es imagen_actual_double (en rango [0,1])
    imagen_para_color_final = imagen_actual_double;

    if ~isempty(filtro_color_seleccionado_nombre)
        % Convertir a uint8 para filtros de color que lo esperan o para consistencia
        img_uint8_para_color = im2uint8(imagen_actual_double); 

        switch filtro_color_seleccionado_nombre % Nombres deben coincidir con DropDown
            case {'Escala_de_Grises', 'Escala de Grises'} % Permitir ambas formas
                if size(img_uint8_para_color, 3) == 3
                    imagen_para_color_final = rgb2gray(img_uint8_para_color);
                else
                    imagen_para_color_final = img_uint8_para_color; % Ya es gris
                end
            case 'Sepia'
                imagen_para_color_final = aplicar_filtro_sepia(img_uint8_para_color); % aplicar_filtro_sepia debe tomar uint8 y devolver uint8
            case 'Invertir'
                imagen_para_color_final = imcomplement(img_uint8_para_color);
            case {'Blanco_y_Negro', 'Blanco y Negro'}
                if size(img_uint8_para_color,3)==3, img_gris = rgb2gray(img_uint8_para_color); else, img_gris = img_uint8_para_color; end
                img_bw = imbinarize(img_gris);
                imagen_para_color_final = uint8(img_bw * 255);
            case 'Calido' % 'Cálido'
                imagen_para_color_final = aplicar_filtro_calido(img_uint8_para_color); % debe tomar uint8 y devolver uint8
            case 'Frio' % 'Frío'
                imagen_para_color_final = aplicar_filtro_frio(img_uint8_para_color); % debe tomar uint8 y devolver uint8
            case {'Ajustar_Saturacion', 'Ajustar Saturación'}
                imagen_para_color_final = aplicar_filtro_hsv(img_uint8_para_color, 2.5); % factor de ejemplo
            case {'Disminuir_Saturacion', 'Disminuir Saturación'}
                imagen_para_color_final = aplicar_filtro_hsv(img_uint8_para_color, 0.3); % factor de ejemplo
            case 'Ninguno'
                % No hacer nada, imagen_para_color_final ya tiene el resultado del kernel (o la original)
                % pero si la original era uint8, devolvemos uint8
                if isa(imagen_original,'uint8') && isa(imagen_para_color_final,'double')
                    imagen_para_color_final = im2uint8(mat2gray(imagen_para_color_final));
                elseif isa(imagen_original,'double') && isa(imagen_para_color_final,'uint8')
                    imagen_para_color_final = im2double(imagen_para_color_final);
                end

            otherwise
                warning('Filtro de color "%s" no reconocido.', filtro_color_seleccionado_nombre);
                % Si no se reconoce, devolver la imagen después del kernel (o la original)
                % Asegurando el tipo de dato de salida consistente con la entrada original
                if isa(imagen_original,'uint8') && isa(imagen_para_color_final,'double')
                    imagen_para_color_final = im2uint8(mat2gray(imagen_para_color_final));
                elseif isa(imagen_original,'double') && isa(imagen_para_color_final,'uint8')
                    imagen_para_color_final = im2double(imagen_para_color_final);
                end
        end
    end

    % Decidir el tipo de salida final. Si la original era uint8, es común devolver uint8.
    % Si era double, devolver double.
    if isa(imagen_original, 'uint8') && isa(imagen_para_color_final, 'double')
        imagen_tratada_final = im2uint8(mat2gray(imagen_para_color_final)); % mat2gray es importante para el rango
    elseif isa(imagen_original, 'double') && isa(imagen_para_color_final, 'uint8')
        imagen_tratada_final = im2double(imagen_para_color_final);
    else
        imagen_tratada_final = imagen_para_color_final; % Ya está en el tipo correcto o consistente
    end
end

% --- Funciones Auxiliares para Filtros de Color (deben tomar uint8 y devolver uint8) ---
function imagen_sepia_out = aplicar_filtro_sepia(imagen_rgb_uint8)
    imagen_double = im2double(imagen_rgb_uint8);
    sepia_matrix = [0.393 0.769 0.189; 0.349 0.686 0.168; 0.272 0.534 0.131];
    imagen_sepia_double = zeros(size(imagen_double));
    if size(imagen_double,3) == 3
        for i = 1:3
            imagen_sepia_double(:,:,i) = imagen_double(:,:,1)*sepia_matrix(i,1) + imagen_double(:,:,2)*sepia_matrix(i,2) + imagen_double(:,:,3)*sepia_matrix(i,3);
        end
        imagen_sepia_out = im2uint8(min(max(imagen_sepia_double,0),1)); % Clamping y conversión
    else
        imagen_sepia_out = imagen_rgb_uint8; % No aplicar a escala de grises
    end
end
function imagen_calida_out = aplicar_filtro_calido(imagen_rgb_uint8)
    imagen_double = im2double(imagen_rgb_uint8);
    if size(imagen_double,3)==3
        imagen_double(:,:,1) = imagen_double(:,:,1) * 1.1; % Rojo
        imagen_double(:,:,2) = imagen_double(:,:,2) * 1.05; % Verde
        imagen_double(:,:,3) = imagen_double(:,:,3) * 0.9;  % Azul
        imagen_calida_out = im2uint8(min(max(imagen_double,0),1));
    else
        imagen_calida_out = imagen_rgb_uint8;
    end
end
function imagen_fria_out = aplicar_filtro_frio(imagen_rgb_uint8)
    imagen_double = im2double(imagen_rgb_uint8);
    if size(imagen_double,3)==3
        imagen_double(:,:,1) = imagen_double(:,:,1) * 0.9;  % Rojo
        imagen_double(:,:,2) = imagen_double(:,:,2) * 1.05; % Verde
        imagen_double(:,:,3) = imagen_double(:,:,3) * 1.2; % Azul
        imagen_fria_out = im2uint8(min(max(imagen_double,0),1));
    else
        imagen_fria_out = imagen_rgb_uint8;
    end
end
function imagen_hsv_out_rgb_uint8 = aplicar_filtro_hsv(imagen_rgb_uint8, saturacion_factor)
    if size(imagen_rgb_uint8,3)==3
        imagen_hsv = rgb2hsv(imagen_rgb_uint8);
        imagen_hsv(:,:,2) = min(max(imagen_hsv(:,:,2) * saturacion_factor, 0), 1); % Ajustar y clampear saturación
        imagen_hsv_out_rgb_uint8 = im2uint8(hsv2rgb(imagen_hsv));
    else
        imagen_hsv_out_rgb_uint8 = imagen_rgb_uint8; % No aplicar a escala de grises
    end
end