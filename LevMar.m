function [C]=LevMar(x, y, C_0)
lambda_0 = 1;
v = 10;
errorTolerance = 0.0001;
gradErrorTolerance = 1e-10;
gradMinDifference = 1e10;
maxIterations = 500;
S = [];

R = errors(x, y, C_0);
S = [S,norm(R)];
convergence=1; %Just needs to be some number bigger than my error tolerance.
C=C_0;
lambda = lambda_0;
iteration = 1;
while (convergence > errorTolerance || ...
        (usedGradientDescent && convergence > gradErrorTolerance) &&...
        ~stop && iteration < maxIterations)
    J=jacobian(x, C);
    JtJdiag = diag(J.'*J).*eye(size(J.'*J));
    dC1=((J.'*J) + lambda * JtJdiag)\J.'*R;
    dC2=(J.'*J + lambda/v * JtJdiag)\J.'*R;
    C1 = C + dC1;
    R1 = errors(x,y,C1);
    C2 = C + dC2;
    R2 = errors(x,y,C2);
    S1 = norm(R1);
    S2 = norm(R2);
    if (S1 < S2 && S1<S(end)) % Default lambda is the best
        usedGradientDescent = false;
        dC = dC1;
        R = R1;
        SE = S1;
        
    elseif (S2 < S1 && S1 < S(end)) % Smaller lambda is preferred
        usedGradientDescent = false;
        dC = dC2;
        R = R2;
        SE = S2;
        lambda = lambda/v;
        
    else % Use gradient descent, keep trying multiples of v for the step size.
       
            S3 = S1;
            dC3 = dC1;
            R3 = R1;
            
            while (S3 > S(end) && lambda < gradMinDifference)
                lambda = lambda * v;
                dC3 = (J.'*J + lambda * JtJdiag)\(J.'*(R));
                C3 = C + dC3;
                R3 = errors(x,y,C3);
                S3 = norm(R3);
            end
            
        
        if(S3 >= S(end))
            stop = true;
        end
        
        usedGradientDescent = true;
        lambda = lambda_0;
        
        dC = dC3;
        R = R3;
        SE = S3;
        
    end
    % Update values
        C = C + dC; 
        S = [S,SE];
        convergence = abs((S(end) - S(end-1))/S(end-1));
        iteration = iteration +1; 
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
