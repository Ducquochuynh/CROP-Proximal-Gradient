function [x, info] = pg_logistic_l1(A, b, lambda, x0, gamma, maxit, tol)
% PG_LOGISTIC_L1:  Proximal Gradient Method for L1-Regularized Logistic Regression
%
%   [x, info] = pg_logistic_l1(A, b, lambda, x0, maxit, tol)
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

% -------------------------------------------------------------------------
if nargin < 4 || isempty(x0), x0 = zeros(size(A,2),1); end
if nargin < 6, maxit = 1000; end
if nargin < 7, tol = 1e-8; end

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
f_hist = zeros(maxit,1);
t_hist = zeros(1, maxit);
res_hist = zeros(1, maxit);
t_start = tic;
phik = obj(x);

% --- Main loop ---
for k = 1:maxit
    x_new = prox_g(x - gamma * grad_f(x), gamma * lambda);
    % Record objective and residual
        phi_new = obj(x_new);
        f_hist(k) = phi_new;
        t_hist(k) = toc(t_start);
        res_hist(k) = abs(phi_new - phik)/(1 + abs(phik));
       
        % Print progress
        fprintf('Iter %4d: f = %.6e, res = %.2e\n', k, f_hist(k), res_hist(k));

        % Check convergence
        if  res_hist(k) < tol
            fprintf('Converged at iteration %d\n', k);
           break;
        end
    x = x_new;
    phik = phi_new;
end
% Trim histories
    k_final = min(k, maxit);
    f_hist = f_hist(1:k_final);
    t_hist = t_hist(1:k_final);
    res_hist = res_hist(1:k_final);

    % Output
    info.f_hist = f_hist;
    info.t_hist = t_hist;
    info.res_hist = res_hist;
    info.k = k_final;
    x = x_new;
    toc(t_start);

end
