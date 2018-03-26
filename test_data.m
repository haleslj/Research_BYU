x = [0, 10, 20, 30, 40]'*1e-3;
y = 2*exp(-x/0.03);

n = 0.05*randn(size(x));

data = y + n;

x2 = linspace(0, 40, 100)'*1e-3;
y2 = 2*exp(-x2/0.03);

figure
hold on
plot(x, data)
plot(x2, y2)
hold off



save('test_data3.mat', 'x', 'data')

data = y;
save('GN_data3.mat', 'x', 'data')