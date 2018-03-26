% Regression Driver
close all
clear all


load('test_data3.mat')
% load('GN_data.mat');
C_0 = [2.5, -1/0.03-5]';
% C_0 = [2, -1/0.03];
% C=GaussNewtonAlgorithm(x,data,C_0);

C2 = LevMar(x, data, C_0); 

% function 
% funobj 

x1 = linspace(x(1), x(end), 100);
best_guess = C2(1)*exp(C2(2)*x1);
figure
hold on 
plot (x, data)
plot(x1,best_guess)
hold off