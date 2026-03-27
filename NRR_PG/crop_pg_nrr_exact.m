function [x, info,acc] = crop_pg_nrr_exact(A, b, lambda, x0, gamma, delta, m, maxit, tol)
% CROP_PG_NRR: CROP Proximal-Gradient Algorithm for Nonnegative
% Ridge Regression (Exact Residual)
%
%   Solves:  minimize 0.5*||A*x - b||^2 + 0.5*lambda*||x||^2   subject to x >= 0
%
%   [x, info, acc] = gcrop_pga_nrr(A,b,lambda,x0,gamma,delta,m,maxit,tol)
%
%   Inputs (defaults used if omitted):
%     x0      - initial guess (default zeros(n,1))
%     gamma   - step size (if empty, set to 1 / L where L = ||A||^2 + lambda
%     delta   - guard parameter (default 1e-4)
%     m       - memory size for Anderson-like combination (default 5)
%     maxit   - maximum PG iterations (default 1000)
%     tol     - stopping tolerance for relative step (default 1e-6)
%
%   Outputs:
%     x       - final iterate
%     info    - struct with fields:
%                x_hist  (stored iterates),
%                f_hist  (objective history),
%                t_hist  (time history),
%                res_hist(relative step norms),
%                k      (number of PG iterations performed)
%     acc     - Number of CROP activated
%


    % ----------------------
    % Argument checks & defaults
    % ----------------------
    if nargin < 9 || isempty(tol), tol = 1e-8; end
    if nargin < 8 || isempty(maxit), maxit = 1000; end
    if nargin < 7 || isempty(m), m = 5; end
    if nargin < 6 || isempty(delta), delta = 1e-4; end

    [mA, n] = size(A);
    if size(b,1) ~= mA, error('Dimension mismatch between A and b'); end

    if nargin < 4 || isempty(x0), x0 = zeros(n,1); end
    x0 = x0(:);

    % Precompute spectral norm if needed for default gamma
    if nargin < 5 || isempty(gamma)
        s = svds(A,1);          % largest singular value of A
        L = s^2 + lambda;  % Lipschitz constant of gradient
        gamma = 1 / L;
    end

    % ----------------------
    % Helpers and precomputations
    % ----------------------
    AtA = A' * A;
    Atb = A' * b;

    % Gradient of smooth part
    gradf = @(x) AtA * x - Atb + lambda * x;

    % Projection onto nonnegative orthant
    proj_pos = @(v) max(v, 0);

    % Objective function
    obj = @(x) 0.5*norm(A*x - b)^2 + 0.5*lambda * norm(x)^2;

    % ----------------------
    % Preallocate histories
    % ----------------------
    x_hist = zeros(n, maxit + 1);    % store iterates x_0 .. x_T
    f_hist = zeros(1, maxit + 1);
    t_hist = zeros(1, maxit + 1);
    res_hist = zeros(1, maxit);      % relative changes after each PG step

    % ----------------------
    % Initialize
% ----------------------
x = x0(:);
phi0 = obj(x);
t_start = tic;

% First iteration (k = 0 → 1)
y1 = x - gamma * gradf(x);
x1 = proj_pos(y1);
phi1 = obj(x1);
grad1 = gradf(x1);
F1 = x1 - gamma * grad1 - y1;

% History allocation
x_hist = zeros(n, maxit+1);
f_hist = zeros(1, maxit+1);
t_hist = zeros(1, maxit+1);
res_hist = zeros(1, maxit);
x_hist(:,1:2) = [x, x1];
f_hist(1:2) = [phi0, phi1];
t_hist(1:2) = toc(t_start);

fprintf('Iter %4d: obj = %.6e, res = %.2e\n',...
    1, f_hist(2), phi1 - phi0)/(1 + abs(phi0));

% Initialize state
x_k = x1; y_k = y1; F_k = F1;
phik = phi1; % Objective value
gradk =  grad1;
Ybuf = []; Hbuf = [];
eps_reg = 1e-12;   % tiny regularizer
k_accept = 0;
% ----------------------
% Main iteration loop
% ----------------------
for k_iter = 1:maxit
    % CROP prediction step
    tilde_y = y_k + F_k;
    tilde_x = proj_pos(tilde_y);
    tilde_F = tilde_x - gamma * gradf(tilde_x) - tilde_y;

    % Update memory buffers (ring buffer style)
    Ybuf = [Ybuf, y_k];
    Hbuf = [Hbuf, F_k];
    if size(Ybuf,2) > m
        Ybuf = Ybuf(:, end-m+1:end);
        Hbuf = Hbuf(:, end-m+1:end);
    end

    % Construct augmented matrices
    Hk = [Hbuf, tilde_F];
    Yk = [Ybuf, tilde_y];

    % Compute Anderson-like combination coefficients
    E = (Hk' * Hk);
    E = (E + E') / 2;   % symmetrize
    s = size(E,1);
    e = ones(s,1);
    z = (E + eps_reg * eye(s)) \ e;
    alpha = z / (e' * z);

    % Form candidate point
    y_check = Yk * alpha;
    x_check = proj_pos(y_check);

    % Guard condition
    eps_k = 1 / (k_iter + 1)^2;
    lhs = obj(x_check);
    rhs = phik + (eps_k - delta) * norm(x_check - x_k)^2;

    if lhs <= rhs
        % Accept accelerated candidate
        x_new = x_check;
        y_new = y_check;
        grad_new = gradf(x_new);
        F_new = x_new - gamma * grad_new - y_new;
        accepted = true;
        k_accept = k_accept + 1;
    else
        % Fallback to plain proximal gradient
        gradk = gradf(x_k);
        y_new = x_k - gamma * gradk;
        x_new = proj_pos(y_new);
        grad_new = gradf(x_new);
        F_new = x_new - gamma * grad_new - y_new;
        accepted = false;
    end
    
    % Record histories
    idx = k_iter + 2;
    x_hist(:,idx) = x_new;
    phi_new = obj(x_new);
    f_hist(idx) = phi_new;
    t_hist(idx) = toc(t_start);
    res_hist(k_iter) = abs(phi_new - phik)/(1 + abs(phik));
    fprintf('Iter %4d: obj = %.6e, res = %.2e, accepted = %d\n', ...
        k_iter+1, f_hist(idx), res_hist(k_iter), double(accepted));

    % Check convergence
    if res_hist(k_iter) < tol 
        break;
    end

    % Update iteration variables
    x_k = x_new; y_k = y_new; F_k = F_new; phik = phi_new;  
    
end

% ----------------------
% Finalize outputs
% ----------------------

% Determine actual number of performed iterations
% If convergence occurred early, k_iter < maxit
if exist('k_iter','var')
    k_final = min(k_iter + 1, maxit);
else
    k_final = 1;
end

% Trim histories safely
K = min(k_final + 1, size(x_hist,2));
x_hist = x_hist(:, 1:K);
f_hist = f_hist(1:K);
t_hist = t_hist(1:K);

% Number of residuals actually computed
num_res = min(k_final, length(res_hist));
res_hist = res_hist(1:num_res);

% Final outputs
x = x_hist(:, end);
info.x_hist = x_hist;
info.f_hist = f_hist;
info.t_hist = t_hist;
info.res_hist = res_hist;
info.k = k_final;
acc = k_accept;

% Safe final print (handle possible empty residual vector)
if ~isempty(res_hist)
    fprintf('Finished at iter %d, final obj = %.6e, final res = %.2e\n', ...
        info.k, f_hist(end), res_hist(end));
else
    fprintf('Finished at iter %d, final obj = %.6e (no residual recorded)\n', ...
        info.k, f_hist(end));
end

toc(t_start);
end

%%%%%%%%%
%%%%%%%%%%%%%%%%%