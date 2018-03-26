function [xopt, fopt, exitflag] = fminun(x0, stoptol, algoflag)
% get function and gradient at starting point
[n,~] = size(x0); % get number of variables
f0 = obj(x0);
grad = gradobj(x0);
gradsize = max(abs(grad));
x = x0;
f = f0;
counter = 0;
fs = f0;
ss = [];
xs = x;
%set starting step length
alpha = 0.5;
alphas = [];
num_evals = [];
if (algoflag == 1) % steepest descent
    s = srchsd(grad);
    ss = [ss, s];
    while (gradsize > stoptol && counter < 1000)
        % find the optimum distance to travel
        [alpha_star, new_evals] = find_alpha_star2(s, x, f, obj, alpha);
        
        % take a step
        x = x + alpha_star*s;
        f = obj(x);
        grad = gradobj(x);
        s = srchsd(grad);
        gradsize = max(abs(grad));
        
        % saving data
        alphas = [alphas;alpha_star];
        fs = [fs; f];
        ss = [ss, s];
        xs = [xs, x];
        num_evals = [num_evals; new_evals];
        counter = counter + 1;
    end
    
elseif (algoflag == 2) % conjugate gradient
    s = -grad;
    ss = [ss, s];
    s_norm = srchsd(-s);
    while (gradsize > stoptol && counter < 500)
        % find the optimum distance to travel
        [alpha_star, new_evals] = find_alpha_star2(s_norm, x, f, obj, alpha);
        
        % take a step
        x = x + alpha_star*s_norm;
        f = obj(x);
        new_grad = gradobj(x);
        gradsize = max(abs(new_grad));
        
        % calculate a new step direction
        beta = (new_grad.'*new_grad)/(grad.'*grad);
        s=-new_grad + beta*s;
        s_norm = srchsd(-s);
        grad=new_grad;
        
        % saving data
        alphas = [alphas; alpha_star];
        fs = [fs; f];
        ss = [ss, s];
        xs = [xs, x];
        num_evals = [num_evals; new_evals];
        
        counter = counter + 1;
    end
    
    
else
    N = eye(length(x));
    s = -N*grad;
    ss = [ss, s];
    while (gradsize > stoptol && counter < 500)
        s_norm=srchsd(-s);
        % find the optimum distance to travel
        [alpha_star, new_evals] = find_alpha_star2(s_norm, x, f, obj, alpha);
        
        % take a step
        x = x + alpha_star*s_norm;
        f=obj(x);
        new_grad = gradobj(x);
        delx=alpha_star*s;
        gamma = new_grad-grad;
        
        % calculateing a new step direction
        N = N + (1 +
        (gamma.'*N*gamma)/(delx.'*gamma))*((delx*delx.')/(delx.'*gamma)) -
        ((delx*gamma.'*N + N*gamma*delx.')/(delx.'*gamma));
        grad = new_grad;
        gradsize = max(abs(new_grad));
        s=-N*grad;
        
        % updating and saving data
        alphas = [alphas; alpha_star];
        fs=[fs;f];
        ss = [ss, s];
        xs = [xs, x];
        num_evals = [num_evals; new_evals];
        counter=counter + 1;
    end
end

xopt = x;
fopt = f;
exitflag = 0;
end

function [f] = obj(C)
%Define the function we are fitting
load('test_data.mat')
func = C(1)*exp(C(2)*x);
f = norm(func - data);
end

function [J] = gradobj(x, C)
J = zeros(length(x), length(C));
J(:, 1) = exp(C(2)*x); %df/dA
J(:, 2) = C(1)*x.*exp(C(2)*x); %df/dB
end

% get steepest descent search direction as a column vector
function [s] = srchsd(grad)
mag = sqrt(grad'*grad);
s = -grad/mag;
end

function [ alpha_star, new_evals ] = find_alpha_star2( s, x0, f0, alpha)
% A line search algorthim to find the optimum distance to travel
% (alpha_star) given direction.
% s - unit vector representing the direction of travel
% x0 - the starting point (independent variable)
% f0 - the starting value (dependent value)
% function object representing the function being minimized.
% alpha - The initial starting distance.
fs = [f0; obj(x0 + alpha*s)];
alphas = [0; alpha];
iterations = 0;
new_evals = 2;
if (fs(2) > fs(1)) % If the step is too large try backtracking. (half the
    step each time)
    while (fs(2) > fs(1))
        alphas = [alphas; alphas(2)/2];
        fs = [fs; obj(x0 + alphas(end)*s)];
        new_evals = new_evals + 1;
        [alphas, I] = sort(alphas);
        fs=fs(I);
        iterations = iterations + 1;
    end
    
    fs=fs(1:3);
    alphas=alphas(1:3);
else % step doubles each round until the function starts to rise again
    alphas = [alphas; alphas(end)*2];
    fs = [fs; obj(x0 + alphas(end)*s)];
    new_evals = new_evals + 1;
    while (fs(end) < fs(end - 1))
        alphas = [alphas; alphas(end)*2];
        fs = [fs; obj(x0 + alphas(end)*s)];
        new_evals = new_evals + 1;
    end
    
    fs = fs(end - 2:end);
    alphas = alphas(end - 2:end);
end
alpha_star = find_min_alpha(alphas,fs);
end


function [ alpha_star ] = find_min_alpha(alpha, f)
%Calculate the minimul of a quadratic given three points. Alphas are the
%independent variable and f is the dependent variable.
Num = f(1)*(alpha(2)^2 - alpha(3)^2) + f(2)*(alpha(3)^2 - alpha(1)^2) +
f(3)*(alpha(1)^2 - alpha(2)^2);
Den = 2*(f(1)*(alpha(2) - alpha(3)) + f(2)*(alpha(3) - alpha(1)) +
f(3)*(alpha(1) - alpha(2)));
alpha_star=Num/Den;
end


