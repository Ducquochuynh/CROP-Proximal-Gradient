function [x, info] = pg_lasso(A, b, lambda, x0, gamma, maxit, tol)
% PG_LASSO: Proximal Gradient Algorithm for LASSO
%
%   [x, info] = pg_lasso(A, b, lambda, x0, maxit, tol)
%
%   Minimizes 0.5*||A*x - b||^2 + lambda*||x||_1 using proximal gradient.
%
%   Inputs:
%     A, b      - problem data
%     lambda    - regularization parameter (>=0)
%     x0        - initial guess (optional)
%     maxit     - maximum iterations (default: 1000)
%     tol       - stopping tolerance (default: 1e-8)
%
%   Outputs:
%     x         - final solution
%     info      - struct with fields:
%                   f_hist (objective history)
%                   t_hist (cumulative time)
%                   res_hist (residual norm history)
%                   k (final iteration)

    % Defaults and checks
    if nargin < 6 || isempty(maxit), maxit = 1000; end
    if nargin < 7 || isempty(tol), tol = 1e-8; end

    [~, n] = size(A);
    if nargin < 4 || isempty(x0), x0 = zeros(n,1); end

    % Precompute constants
    AtA = A' * A;
    Atb = A' * b;

    % Objective and proximal functions
    obj = @(x) 0.5 * norm(A*x - b)^2 + lambda * sum(abs(x));
    prox = @(v, t) sign(v) .* max(abs(v) - t, 0);
    
    % Initialize
    x = x0;
    phik = obj(x);
    f_hist = zeros(1, maxit);
    t_hist = zeros(1, maxit);
    res_hist = zeros(1, maxit);
    t_start = tic;

    % Start the loop
    for k = 1:maxit
        grad = AtA*x - Atb;
        x_new = prox(x - gamma * grad, gamma * lambda);

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
