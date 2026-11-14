function [x, info, acc] = crop_pg_lasso_exact(A, b, lambda, x0, gamma, delta, m, maxit, tol, c, print_every)
% Optimized CROP-PG for LASSO: 0.5*||A*x-b||^2 + lambda*||x||_1.
% Residual variant: EXACT.
% Original calling convention preserved; optional c (default 1) and
% print_every (default 0: summary only, 1: every iteration).
% c=Inf turns OFF the consistency safeguard for baseline comparisons.
% Existing objective safeguard and objective-change stopping are preserved.
%
% Extra diagnostics: info.accept_hist, info.consistency_hist,
% info.bound_hist, info.actual_error_hist, info.reject_objective,
% info.reject_consistency, info.accept_rate, info.gradient_mode.

if nargin < 11 || isempty(print_every), print_every = 0; end
if nargin < 10 || isempty(c), c = 1; end
if nargin < 9 || isempty(tol), tol = 1e-8; end
if nargin < 8 || isempty(maxit), maxit = 1000; end
if nargin < 7 || isempty(m), m = 5; end
if nargin < 6 || isempty(delta), delta = 1e-4; end

if ~isscalar(c) || ~isreal(c) || isnan(c) || c <= 0
    error('c must be positive; c=Inf disables the consistency safeguard.');
end
if ~isscalar(m) || ~isfinite(m) || m < 1 || m ~= floor(m)
    error('m must be a positive integer.');
end
if ~isscalar(maxit) || ~isfinite(maxit) || maxit < 1 || maxit ~= floor(maxit)
    error('maxit must be a positive integer.');
end
if ~isscalar(tol) || ~isfinite(tol) || tol < 0
    error('tol must be a nonnegative finite scalar.');
end
if ~isscalar(print_every) || ~isfinite(print_every) || print_every < 0 || print_every ~= floor(print_every)
    error('print_every must be a nonnegative integer.');
end
[nobs, n] = size(A);
b = b(:);
if numel(b) ~= nobs, error('Dimension mismatch between A and b.'); end
if nargin < 4 || isempty(x0), x0 = zeros(n,1); end
x0 = x0(:);
if numel(x0) ~= n, error('Dimension mismatch between A and x0.'); end
if nargin < 5 || isempty(gamma)
    L = svds(A,1)^2; % preserve the original default stepsize
    if L <= 0, error('A must not be the zero matrix when gamma is omitted.'); end
    gamma = 1 / L;
end
if ~isscalar(gamma) || ~isfinite(gamma) || gamma <= 0
    error('gamma must be a positive finite scalar.');
end

% For tall/narrow problems use Gram products; for wide/large problems
% avoid forming the potentially huge dense n-by-n Gram matrix.
use_gram = n <= min(nobs,1200);
if use_gram
    G = A' * A;
    g = A' * b;
    gradf = @(v) G*v - g;
    gradient_mode = 'Gram';
else
    gradf = @(v) A'*(A*v-b);
    gradient_mode = 'matrix-free';
end
prox_l1 = @(v) sign(v).*max(abs(v)-gamma*lambda,0);
obj = @(v) 0.5*sum(abs(A*v-b).^2) + lambda*sum(abs(v));

start_time = tic;
x_old = x0;
phi0 = obj(x_old);
y1 = x_old - gamma*gradf(x_old);
x_k = prox_l1(y1);
phik = obj(x_k);
y_k = y1;
F_k = x_k - gamma*gradf(x_k) - y_k;
x_prev = x_old;

% maxit means total iterations, including the initial PG step.
x_hist = zeros(n,maxit+1);
f_hist = zeros(1,maxit+1);
t_hist = zeros(1,maxit+1);
res_hist = zeros(1,max(0,maxit-1)); % original convention: after initial PG
accept_hist = false(1,max(0,maxit-1));
consistency_hist = nan(1,max(0,maxit-1));
bound_hist = nan(1,max(0,maxit-1));
actual_error_hist = nan(1,max(0,maxit-1));
x_hist(:,1:2) = [x_old,x_k];
f_hist(1:2) = [phi0,phik];
t_hist(1:2) = [0,toc(start_time)];
initial_res = abs(phik-phi0)/(1+abs(phi0));

% Bounded history memory (only m columns).
Ybuf = zeros(n,0);
Hbuf = zeros(n,0);
eps_reg = 1e-12; % retain original regularization value
acc = 0;
reject_objective = 0;
reject_consistency = 0;
num_extra = 0;
for j = 1:(maxit-1)
    % Predictor and associated residual. Note: the inexact variant must
    % retain Hk*alpha as its accepted residual approximation.
    tilde_y = y_k + F_k;
    tilde_x = prox_l1(tilde_y);
    tilde_F = tilde_x - gamma*gradf(tilde_x) - tilde_y;

    Ybuf = [Ybuf,y_k];
    Hbuf = [Hbuf,F_k];
    if size(Ybuf,2) > m
        Ybuf(:,1) = [];
        Hbuf(:,1) = [];
    end
    Hk = [Hbuf,tilde_F];
    Yk = [Ybuf,tilde_y];

    % SPD regularized least squares; small (m+1)-by-(m+1) solve.
    B = Hk'*Hk;
    B = (B+B')/2;
    q = size(B,1);
    e = ones(q,1);
    w = (B+eps_reg*eye(q))\e;
    alpha = w/(e'*w);
    y_check = Yk*alpha;
    x_check = prox_l1(y_check);

    % The objective value of an accepted candidate is reused below.
    phi_check = obj(x_check);
    eps_k = 1/(j+1)^2;
    step_sq = sum(abs(x_check-x_k).^2);
    pass_objective = isfinite(phi_check) && ...
        phi_check <= phik+(eps_k-delta)*step_sq;

    % z_k equals y_k+F_k only for EXACT residuals. For the inexact
    % residual variant recompute it from the true gradient.
    z_k = y_k + F_k;
    consistency_error = norm(y_check-z_k);
    consistency_bound = c*norm(x_k-x_prev);
    pass_consistency = isinf(c) || ...
        (isfinite(consistency_error) && consistency_error <= consistency_bound);
    accepted = pass_objective && pass_consistency;
    accept_hist(j) = accepted;
    consistency_hist(j) = consistency_error;
    bound_hist(j) = consistency_bound;

    if ~pass_objective, reject_objective = reject_objective+1; end
    if pass_objective && ~pass_consistency
        reject_consistency = reject_consistency+1;
    end

    if accepted
        y_new = y_check;
        x_new = x_check;
        phi_new = phi_check;
        F_new = x_new - gamma*gradf(x_new) - y_new;
        acc = acc+1;
    else
        y_new = z_k;
        x_new = prox_l1(y_new);
        phi_new = obj(x_new);
        F_new = x_new - gamma*gradf(x_new) - y_new;
    end

    % Actual inconsistency: 0 on fallback, candidate error on acceptance.
    actual_error_hist(j) = double(accepted)*consistency_error;
    num_extra = j;
    ii = j+2;
    x_hist(:,ii) = x_new;
    f_hist(ii) = phi_new;
    t_hist(ii) = toc(start_time);
    res_hist(j) = abs(phi_new-phik)/(1+abs(phik));
    if print_every > 0 && (mod(j,print_every)==0 || res_hist(j)<tol)
        fprintf('Iter %5d  phi %.10e  change %.3e  CROP %d\n', ...
            j+1,phi_new,res_hist(j),accepted);
    end
    if res_hist(j) < tol
        break;
    end
    x_prev = x_k;
    x_k = x_new;
    y_k = y_new;
    F_k = F_new;
    phik = phi_new;
end

n_iter = num_extra+1;
n_saved = n_iter+1;
x_hist = x_hist(:,1:n_saved);
f_hist = f_hist(1:n_saved);
t_hist = t_hist(1:n_saved);
res_hist = res_hist(1:num_extra);
info = struct();
info.x_hist = x_hist;
info.f_hist = f_hist;
info.t_hist = t_hist;
info.res_hist = res_hist;
info.k = n_iter;
info.c = c;
info.initial_res = initial_res;
info.accept_hist = accept_hist(1:num_extra);
info.consistency_hist = consistency_hist(1:num_extra);
info.bound_hist = bound_hist(1:num_extra);
info.actual_error_hist = actual_error_hist(1:num_extra);
info.reject_objective = reject_objective;
info.reject_consistency = reject_consistency;
info.accept_rate = acc/max(num_extra,1);
info.gradient_mode = gradient_mode;
info.runtime = toc(start_time);
x = x_hist(:,end);
if isempty(res_hist), final_res = initial_res; else, final_res = res_hist(end); end
fprintf('crop_pg_lasso_exact_modified: iter=%d, phi=%.9e, change=%.2e, accepted=%d/%d (%.1f%%), time=%.3fs\n', ...
    n_iter,f_hist(end),final_res,acc,num_extra,100*info.accept_rate,info.runtime);
end
