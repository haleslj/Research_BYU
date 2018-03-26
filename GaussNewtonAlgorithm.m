function [C]=GaussNewtonAlgorithm(x,y,C_0)

damping = 1;
errorTolerance = 0.0001;
gradErrorTolerance = 1e-10;
gradMinDifference = 1e30;
maxIterations = 100;
S = [];

R = errors(x, y, C_0);
S = [S,norm(R)];
convergence=1; %Just needs to be some number bigger than my error tolerance.
C=C_0;
while (convergence > errorTolerance)
    J=jacobian(x, C);
    dC=(J.'*J)\J.'*R;
    C = C + dC';
    R = errors(x,y,C);
    S = [S,norm(R)];
    convergence = abs((S(end) - S(end-1))/S(end-1));
end

end

function [f] = funobj(x, C)
%Define the function we are fitting
f = C(1)*exp(C(2)*x);
end

function [J] = jacobian(x, C)
J = zeros(length(x), length(C));
J(:, 1) = exp(C(2)*x); %df/dA
J(:, 2) = C(1)*x.*exp(C(2)*x); %df/dB
end

function [R] = errors(x, y, C)
R = y-funobj(x, C);
end
