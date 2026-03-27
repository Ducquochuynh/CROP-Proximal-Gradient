function [x, info, acc] = aa_pg_lasso(A, b, lambda, x0, gamma, m, maxit, tol, opts)
% AA_PG: Anderson-Acceleration Proximal-Gradient for LASSO
%
% [x, info, acc] = aa_pg_lasso(A,b,lambda,x0,gamma,delta,m,maxit,tol,opts)
%
% Inputs:
%   A, b         : problem data
%   lambda       : l1-regularization parameter
%   x0           : initial guess (default zeros)
%   gamma        : stepsize (default 1/L)
%   delta        : reserved (unused)
%   m            : Anderson memory size (default 5)
%   maxit        : max iterations (default 1000)
%   tol          : stopping tolerance (default 1e-8)
%   opts         : optional struct with fields
%       .verbose : true/false (default true)
%       .reg_eps : small regularization for AA (default 1e-12)
%
% Outputs:
%   x    : final iterate
%   info : struct with fields x_hist, f_hist, t_hist, res_hist, k
%   acc  : number of AA activated

if nargin < 1, error('Provide A'); end
if nargin < 2, error('Provide b'); end
if nargin < 3 || isempty(lambda), lambda = 0; end
if nargin < 4 || isempty(x0), x0 = zeros(size(A,2),1); end
if nargin < 5, gamma = []; end
if nargin < 6 || isempty(m), m = 5; end
if nargin < 7 || isempty(maxit), maxit = 1000; end
if nargin < 8 || isempty(tol), tol = 1e-8; end
if nargin < 9, opts = struct(); end
if ~isfield(opts,'verbose'), opts.verbose = true; end
if ~isfield(opts,'reg_eps'), opts.reg_eps = 1e-12; end

% Problem setup
[mA, n] = size(A);
if size(b,1) ~= mA, error('Dimension mismatch'); end
x0 = x0(:);
if numel(x0) ~= n, error('x0 must be length %d', n); end

% Default gamma = 1 / L where L = ||A||_2^2
if isempty(gamma)
    try
        s = svds(A,1); L = s^2;
    catch
        L = norm(A)^2;
    end
    gamma = 1 / max(L, eps);
end

% Gradient and objective functions
AtA = A'*A; Atb = A'*b;
gradf = @(x) AtA*x - Atb;
obj_f = @(x) 0.5*norm(A*x - b)^2;
obj_g = @(x) lambda*norm(x,1);
obj = @(x) obj_f(x) + obj_g(x);

% Preallocate history
x_hist = zeros(n, maxit+2);
f_hist = zeros(1, maxit+2);
t_hist = zeros(1, maxit+2);
res_hist = zeros(1, maxit);
t_start = tic;

% Initialization
x = x0;
y = x0; 
y1 = x - gamma*gradf(x);
x1 = prox_l1(y1, gamma*lambda);
g0 = y1;
phi0 = obj(x);
phi1 = obj(x1);

% Circular buffers for Anderson memory
Rbuf = zeros(n, m+1);  % residuals
Gbuf = zeros(n, m+1);  % candidates
buf_count = 0;          % number of stored vectors
buf_idx = 0;            % next insert position

% Store first residual
r0 = g0 - y;
buf_idx = mod(buf_idx, m+1) + 1;
Rbuf(:, buf_idx) = r0;
Gbuf(:, buf_idx) = g0;
buf_count = min(buf_count + 1, m+1);

% Record history for x0 and x1
x_hist(:,1) = x0; f_hist(1) = phi0; t_hist(1) = toc(t_start);
x_hist(:,2) = x1; f_hist(2) = phi1; t_hist(2) = toc(t_start);

if opts.verbose
    fprintf('Iter %4d: obj = %.6e, res = %.2e\n',...
        1, f_hist(2), abs(phi1 - phi0)/(1+abs(phi0)));
end

x_k = x1; y_k = y1;  % current iterate and candidate
fk = obj_f(x_k);
phik = phi1;
k_final = 1;
k_accept = 0;

% Main loop
for iter = 1:maxit
    % Gradient step and residual
    gradk = gradf(x_k);
    gk = x_k - gamma*gradk;
    rk = gk - y_k;

    % Update circular buffer
    buf_idx = mod(buf_idx, m+1) + 1;
    Rbuf(:, buf_idx) = rk;
    Gbuf(:, buf_idx) = gk;
    buf_count = min(buf_count + 1, m+1);

    % Extract last (m+1) iterates in time order
    if buf_count == m+1
        idx = mod((buf_idx-(m+1)+1 : buf_idx)-1, m+1) + 1;
    else
        idx = 1:buf_count;
    end
    Rk = Rbuf(:, idx); 
    Gk = Gbuf(:, idx);
    s = size(Rk,2);

    % Compute Anderson coefficients (normalized)
    H = Rk'*Rk;
    H = (H + H')/2;       % ensure symmetric
    H_reg = H + opts.reg_eps*eye(s);
    e = ones(s,1);
    z = H_reg \ e;
    alpha = z / (e'*z);

    % AA candidate and proximal step
    y_check = Gk*alpha;
    x_check = prox_l1(y_check, gamma*lambda);

    % Guarded acceptance test
    f_check = obj_f(x_check);
    lhs = f_check;
    rhs = fk - 1e-4*(gamma/2)*(norm(gradk)^2);
    
    if lhs <= rhs
        x_new = x_check;
        y_new = y_check;
        f_new = f_check;
        accepted = true;
        k_accept = k_accept + 1;
    else
        y_new = gk;
        x_new = prox_l1(y_new, gamma*lambda);
        f_new = obj_f(x_new);
        accepted = false;
    end

    % Record history
    col = iter + 2;
    x_hist(:, col) = x_new;
    phi_new = f_new + obj_g(x_new);
    f_hist(col) = phi_new;
    t_hist(col) = toc(t_start);
    res = abs(phi_new - phik)/(1 + abs(phik));
    res_hist(iter) = res;

    if opts.verbose
        fprintf('Iter %4d: obj = %.6e, res = %.2e, accepted = %d\n',...
            iter+1, f_hist(col), res, double(accepted));
    end

    % Check stopping
    if res < tol
        k_final = iter + 1;
        break;
    end

    % Prepare next iteration
    x_k = x_new; 
    y_k = y_new;
    fk = f_new;
    phik = phi_new;
    k_final = iter + 1;
end

% Trim history
K = min(k_final+1, size(x_hist,2));
x_hist = x_hist(:,1:K);
f_hist = f_hist(1:K);
t_hist = t_hist(1:K);
res_hist = res_hist(1:max(0,k_final-1));

% Outputs
x = x_hist(:, end);
info.x_hist = x_hist;
info.f_hist = f_hist;
info.t_hist = t_hist;
info.res_hist = res_hist;
info.k = k_final;
acc = k_accept;
if opts.verbose
    if ~isempty(res_hist)
        fprintf('Finished at iter %d, final obj = %.6e, final res = %.2e\n',...
            info.k, f_hist(end), res_hist(end));
    else
        fprintf('Finished at iter %d, final obj = %.6e, (no residual recorded)\n',...
            info.k, f_hist(end));
    end
end
toc(t_start);

end
%------------------------------
% Soft-thresholding proximal operator
function z = prox_l1(v, thresh)
    z = sign(v) .* max(abs(v) - thresh, 0);
end
