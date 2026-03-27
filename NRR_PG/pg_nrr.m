function [x, info] = pg_nrr(A, b, lambda, x0, gamma, maxit, tol)
% PG_NRR: Proximal Gradient Algorithm for Nonnegative Ridge Regression
%
%   Solves
%       minimize f(x) = 0.5*||A*x - b||^2 + 0.5*lambda*||x||^2
%       subject to x >= 0.
%
%   [x, info] = pg_nrr(A, b, lambda, x0, maxit, tol)
%
%   Inputs:
%     A, b      - problem data
%     lambda    - regularization parameter (>=0)
%     x0        - initial guess (default: zeros)
%     maxit     - maximum iterations (default: 1000)
%     tol       - stopping tolerance for relative change (default: 1e-8)
%
%   Outputs:
%     x         - final solution
%     info      - struct with fields:
%                   f_hist   (objective value history)
%                   res_hist (step norm history)
%                   t_hist   (cumulative time)
%                   k        (final iteration)
%

    %-------------------------------%
    % Default parameter settings
    %-------------------------------%
    if nargin < 6 || isempty(tol), tol = 1e-8; end
    if nargin < 5 || isempty(maxit), maxit = 1000; end
    [~, n] = size(A);
    if nargin < 4 || isempty(x0), x0 = zeros(n,1); end

    %-------------------------------%
    % Precomputations
    %-------------------------------%
    AtA = A' * A;
    Atb = A' * b;

    % Objective function
    obj = @(x) 0.5*norm(A*x - b)^2 + 0.5*lambda*norm(x)^2;
    % Gradient of smooth part
    gradf = @(x) (AtA * x - Atb) + lambda * x;

    % Projection onto nonnegative orthant
    proj_pos = @(v) max(v, 0);

    %-------------------------------%
    % Initialization
    %-------------------------------%
    x = x0;
    f_hist = zeros(1, maxit);
    res_hist = zeros(1, maxit);
    t_hist = zeros(1, maxit);

    t_start = tic;
    phik = obj(x);
    %-------------------------------%
    % Main loop
    %-------------------------------%
    for k = 1:maxit
        % Gradient step
        grad = gradf(x);
        x_new = proj_pos(x - gamma * grad);

        % Record progress
        phi_new = obj(x_new);
        f_hist(k) = phi_new;
        res_hist(k) = abs(phi_new - phik)/(1+abs(phik)); % relative residual
        t_hist(k) = toc(t_start);

        % Display progress
        fprintf('Iter %4d: f = %.6e, res = %.2e\n', k, f_hist(k), res_hist(k));

        % Convergence check
        if res_hist(k) < tol 
            fprintf('Converged at iteration %d\n', k);
            break;
        end

        x = x_new;
        phik = phi_new;
    end

    %-------------------------------%
    % Output
    %-------------------------------%
    k_final = min(k, maxit);
    info.f_hist = f_hist(1:k_final);
    info.res_hist = res_hist(1:k_final);
    info.t_hist = t_hist(1:k_final);
    info.k = k_final;
    x = x_new;

    toc(t_start);
end
