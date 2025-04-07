%Realizado por Juan Esteban Fuentes, Laura Latorre y Duvan Santiago Matallana
%Limpieza
close all;
clear;
clc;
%Patrones de entrada
x=readmatrix("entradas.txt");
%Salida esperada
yd=readmatrix("salidas.txt");
%Definir tamaños de entrada y salida:
[patrones, tam_x] = size(x);
tam_y = length(yd);
%definir número de neuronas de la capa oculta
n_oculta = 2;
%Definir tasa de aprendizaje y precisión de la red
alpha = 0.05;
precision = 0.000001;
% Inicializar pesos y sesgos (uno para cada capa)
%pesos de capa entrada a oculta
W1 = 2*rand(tam_x, n_oculta)-1;
%pesos de capa oculta a salida
W2 = 2*rand(n_oculta, tam_y);
%Sesgos de capa oculta y de salida
b1 = rand(1,n_oculta);
b2 = rand(1,tam_y);
%función de activación sigmoidal
sigmoidal = @(x) 1./(1+exp(-x));
%Derivada de la función para la retropopagación:
sigmoidal_deriv = @(a) a .* (1 - a);
%Inicialización de épocas
epocas=0;
errorActual = 1000000;
Eh = [];
fprintf('tamaño: %d', patrones);

while(errorActual>precision)
    epocas = epocas +1;
    errorTotal = 0;
    for i = 1:patrones
        % ---------------------------
        % Paso 1: Propagación hacia adelante
        % ---------------------------
        a1 = x(i,:);                      % Entrada (fila 1 x nInputs)
        z1 = a1 * W1 + b1;                 % Suma ponderada en la capa oculta
        a2 = sigmoidal(z1);                  % Activación de la capa oculta
        
        z2 = a2 * W2 + b2;                 % Suma ponderada en la capa de salida
        a3 = sigmoidal(z2);
        % ---------------------------
        % Paso 2: Cálculo del error
        % ---------------------------
        for j = 1:tam_y
            error = yd(j,:) - a3;              % Error en la salida
            errorTotal = errorTotal + sum(error.^2);% Activación de la capa de salida
        end
        % ---------------------------
        % Paso 3: Retropropagación del error
        % ---------------------------
        % Delta en la capa de salida:
        delta1 = error .* sigmoidal_deriv(a3);
        
        % Delta en la capa oculta:
        delta2 = (delta1 * W2') .* sigmoidal_deriv(a2);
       % ---------------------------
        % Paso 4: Actualización de pesos y sesgos
        % ---------------------------
        % Actualización para la capa de salida:
        W2 = W2 + alpha * (a2' * delta1);
        b2 = b2 + alpha * delta1;
        
        % Actualización para la capa oculta:
        W1 = W1 + alpha * (a1' * delta2);
        b1 = b1 + alpha * delta2; 
    end
    % Error medio de la época actual
    errorActual = errorTotal / patrones;  
    Eh = [Eh;errorActual];
end
disp(W1);
fprintf('Entrenamiento completado en %d épocas con error %f\n', epocas, errorActual);
% Visualización del error durante el entrenamiento
figure;
plot(Eh, 'LineWidth', 2);
title('Error a lo largo de las épocas');
xlabel('Época');
ylabel('Error medio');
grid on;