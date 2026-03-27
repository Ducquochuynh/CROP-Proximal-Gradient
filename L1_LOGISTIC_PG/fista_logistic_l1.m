function [x, info] = fista_logistic_l1(A, b, lambda, x0, gamma, maxit, tol)
% FISTA_LOGISTIC_L1:  FISTA for L1-Regularized Logistic Regression
%
%   [x, info] = fista_logistic_l1(A, b, lambda, x0, gamma, maxit, tol)
%
%   Solves:  minimize (1/m) * sum(log(1 + exp(-b_i * a_i' * x))) + lambda * ||x||_1
%
%   Inputs:
%     A       : m x n data matrix
%     b       : m x 1 label vector with entries in {-1, +1}
%     lambda  : regularization parameter
%     x0      : initial guess (default = zeros(n,1))
%     gamma   : stepsize (default = 1/L)
%     maxit   : maximum iterations (default = 1000)
%     tol     : stopping tolerance (default = 1e-8)
%
%   Outputs:
%     x       : final solution
%     info    : structure with convergence information:
%               info.f_hist  - objective values
%               info.iter    - number of iterations
%               info.time    - elapsed time (seconds)
%               info.final_obj - final objective value
%
% -------------------------------------------------------------------------
if nargin < 4 || isempty(x0), x0 = zeros(size(A,2),1); end
if nargin < 5, maxit = 1000; end
if nargin < 6, tol = 1e-8; end

[m, ~] = size(A);
sigmoid = @(z) 1 ./ (1 + exp(-z));

% --- Objective, gradient, prox operators ---
f  = @(x) (1/m) * sum(log(1 + exp(-b .* (A*x))));
grad_f = @(x) -(A' * (b .* sigmoid(-b .* (A*x)))) / m;
g  = @(x) lambda * norm(x,1);
prox_g = @(v, t) sign(v) .* max(abs(v) - t, 0);
obj = @(x) f(x) + g(x);

% --- Initialize ---
    x = x0;
    y = x0;
    t = 1;

    f_hist = zeros(1, maxit);
    t_hist = zeros(1, maxit);
    res_hist = zeros(1, maxit);

    t_start = tic;
    phik = obj(x);

    for k = 1:maxit
        % Gradient step
        grad_y = grad_f(y);
        x_new = prox_g(y - gamma * grad_y, gamma * lambda);
        
        % FISTA momentum update
        t_new = 0.5 * (1 + sqrt(1 + 4 * t^2));
        y = x_new + ((t - 1) / t_new) * (x_new - x);

        % Record objective and residual
        phi_new = obj(x_new);
        f_hist(k) = phi_new;
        t_hist(k) = toc(t_start);
        res_hist(k) = abs(phi_new - phik)/(1 + abs(phik));

        % Display progress
        fprintf('Iter %4d: f = %.6e, res = %.2e\n', k, f_hist(k), res_hist(k));

        % Convergence check
        if res_hist(k) < tol 
            fprintf('Converged at iteration %d\n', k);
            break;
        end

        % Update for next iteration
        x = x_new;
        phik = phi_new;
        t = t_new;
    end

    % Trim histories
    k_final = min(k, maxit);
    f_hist = f_hist(1:k_final);
    t_hist = t_hist(1:k_final);
    res_hist = res_hist(1:k_final);

    % Output struct
    info.f_hist = f_hist;
    info.t_hist = t_hist;
    info.res_hist = res_hist;
    info.k = k_final;
    x = x_new;
    toc(t_start);
end
