function imagen_tratada = Aplicar_filtros(imagen_original, kernel_seleccionado, filtro_color_seleccionado)
% APLICAR_FILTROS Aplica kernels y filtros de color a una imagen.
%
%   IMAGEN_TRATADA = APLICAR_FILTROS(IMAGEN_ORIGINAL, KERNEL_SELECCIONADO, FILTRO_COLOR_SELECCIONADO)
%
%   Entradas:
%       IMAGEN_ORIGINAL        Matriz de la imagen original (uint8).
%       KERNEL_SELECCIONADO    String con el nombre del kernel a aplicar.
%                              Puede ser: 'Enfoque Sharpen', 'Deteccion de bordes Laplacian',
%                                         'Deteccion de Bordes Fuerte', 'Desenfoque Box Blur',
%                                         'Repujado Emboss', 'Prewitt_X', 'Prewitt_Y',
%                                         'Sobel_X', 'Sobel_Y', 'Desenfoque Gaussiano',
%                                         'Desenfoque Medio', 'Deteccion de Bordes Laplaciano',
%                                         'Realce Sharpen', 'Relieve Emboss', 'Prewitt_X',
%                                         'Prewitt_Y', 'Canny' (Nota: Canny se aplica diferente).
%       FILTRO_COLOR_SELECCIONADO String con el nombre del filtro de color a aplicar.
%                                  Puede ser: 'Escala de Grises', 'Sepia', 'Invertir',
%                                              'Blanco y Negro', 'Cálido', 'Frío'.a
%
%   Salida:
%       IMAGEN_TRATADA         Matriz de la imagen tratada (double o uint8).
imagen_tratada = imagen_original; % Inicializar con la imagen original
% --- Definición de Kernels ---
kernels = struct();
% Kernels 3x3
kernels.('Enfoque_Sharpen') = [0 -1 0; -1 5 -1; 0 -1 0];
kernels.('Deteccion_de_bordes_Laplaciano') = [0 1 0; 1 -4 1; 0 1 0];
kernels.('Deteccion_de_Bordes_Fuerte') = [-1 -1 -1; -1 8 -1; -1 -1 -1];
kernels.('Desenfoque_Box_Blur') = [1 1 1;1 1 1;1 1 1];
kernels.('Repujado_Emboss_1') = [-2 -1 0; -1 1 1; 0 1 2];
kernels.('Repujado_Emboss_2') = [-1 -1 0; -1 1 1; 0 1 1];
kernels.('Prewitt_X') = [-1 0 1; -1 0 1; -1 0 1];
kernels.('Prewitt_Y') = [-1 -1 -1; 0 0 0; 1 1 1];
kernels.('Sobel_X') = [-1 0 1; -2 0 2; -1 0 1];
kernels.('Sobel_Y') = [-1 -2 -1; 0 0 0; 1 2 1];
kernels.('Realce_Laplaciano') = [-1 -1 -1; -1 9 -1; -1 -1 -1];
% Kernels 5x5
kernels.('Desenfoque_Gaussiano') = fspecial('gaussian', 5, 1); % Sigma = 1 (ajustable)
kernels.('Desenfoque_Medio') = ones(5)/25;
kernels.('Deteccion_de_Bordes_Laplaciano_5x5') = [1 1 1 1 1; 1 1 1 1 1; 1 1 -24 1 1; 1 1 1 1 1; 1 1 1 1 1];
kernels.('Realce_Sharpen_5x5') = [ -1 -1 -1 -1 -1; -1  2  2  2 -1; -1  2  8  2 -1; -1  2  2  2 -1; -1 -1 -1 -1 -1] / 8;
kernels.('Relieve_Emboss_5x5') = [-2 -1 0 1 2; -2 -1 0 1 2; -2 -1 1 1 2; -2 -1 0 1 2; -2 -1 0 1 2];
kernels.('Prewitt_X_5x5') = repmat([-2 -1 0 1 2], 5, 1) / 10;
kernels.('Prewitt_Y_5x5') = repmat([-2; -1; 0; 1; 2], 1, 5) / 10;
% --- Aplicar Kernel ---
if isfield(kernels, kernel_seleccionado)
    if strcmp(kernel_seleccionado, 'Canny')
        imagen_gris = rgb2gray(imagen_original);
        imagen_tratada = edge(imagen_gris, 'canny');
        imagen_tratada = uint8(imagen_tratada * 255); % Convertir a uint8 para visualización
    else
        kernel = kernels.(kernel_seleccionado);
        imagen_tratada = imfilter(imagen_original, kernel, 'replicate');
    end
end
% --- Aplicar Filtro de Color ---
if ~isempty(filtro_color_seleccionado)
    switch filtro_color_seleccionado
        case 'Escala de Grises'
            imagen_tratada = rgb2gray(imagen_tratada);
        case 'Sepia'
            imagen_tratada = aplicar_filtro_sepia(imagen_tratada);
        case 'Invertir'
            imagen_tratada = imcomplement(imagen_tratada);
        case 'Blanco y Negro'
            imagen_gris = rgb2gray(imagen_tratada);
            imagen_tratada = imbinarize(imagen_gris);
            imagen_tratada = uint8(imagen_tratada * 255);
        case 'Cálido'
            imagen_tratada = aplicar_filtro_calido(imagen_tratada);
        case 'Frío'
            imagen_tratada = aplicar_filtro_frio(imagen_tratada);
    end
end
end
% --- Funciones Auxiliares para Filtros de Color ---
function imagen_sepia = aplicar_filtro_sepia(imagen_rgb)
imagen_double = im2double(imagen_rgb);
sepia_matrix = [0.393 0.769 0.189;
    0.349 0.686 0.168;
    0.272 0.534 0.131];
imagen_sepia_double = zeros(size(imagen_double));
for i = 1:3
    imagen_sepia_double(:,:,i) = sum(imagen_double .* repmat(reshape(sepia_matrix(i,:), [1 1 3]), size(imagen_double,1), size(imagen_double,2)), 3);
end
imagen_sepia = im2uint8(imagen_sepia_double);
end
function imagen_calida = aplicar_filtro_calido(imagen_rgb)
imagen_double = im2double(imagen_rgb);
imagen_calida = imagen_double;
imagen_calida(:,:,1) = min(1, imagen_calida(:,:,1) * 1.1); % Rojo
imagen_calida(:,:,2) = min(1, imagen_calida(:,:,2) * 1.05); % Verde
imagen_calida(:,:,3) = max(0, imagen_calida(:,:,3) * 0.9);  % Azul
imagen_calida = im2uint8(imagen_calida);
end
function imagen_fria = aplicar_filtro_frio(imagen_rgb)
imagen_double = im2double(imagen_rgb);
imagen_fria = imagen_double;
imagen_fria(:,:,1) = max(0, imagen_fria(:,:,1) * 0.9);  % Rojo
imagen_fria(:,:,2) = min(1, imagen_fria(:,:,2) * 1.05); % Verde
imagen_fria(:,:,3) = min(1, imagen_fria(:,:,3) * 1.2); % Azul
imagen_fria = im2uint8(imagen_fria);
end