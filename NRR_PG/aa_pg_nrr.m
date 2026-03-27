function [x, info,acc] = aa_pg_nrr(A, b, lambda, x0, gamma, m, maxit, tol, opts)
% AA_PG_NRR: Anderson-Acceleration Proximal-Gradient Algorithm for
% Nonnegative Ridge Regression
%
%   Solves: minimize 0.5*||A*x - b||^2 + 0.5*lambda*||x||^2  subject to x >= 0
%
%   [x, info, acc] = aa_pg_nrr(A,b,lambda,x0,gamma,m,maxit,tol,opts)
%
%   Inputs:
%     A, b    - problem data
%     lambda  - l2 regularization parameter (>=0)
%     M       - positive scaling constant (required)
%     x0      - initial guess (default zeros)
%     gamma   - stepsize (default 1/L where L = ||A||^2 + lambda
%     m       - Anderson memory size (default 5)
%     maxit   - max iterations (default 1000)
%     tol     - stopping tolerance (default 1e-6)
%     opts    - struct with optional fields:
%                 .verbose (true/false), .reg_eps (regularizer for AA)
%
%   Outputs:
%     x         - final solution
%     info      - struct with fields:
%                   f_hist   (objective value history)
%                   res_hist (step norm history)
%                   t_hist   (cumulative time)
%                   k        (final iteration)
%     acc       - Number of AA activated
%


    % ----------------------
    % Input checks & defaults
    % ----------------------
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

    % Problem dims & validate
    [mA, n] = size(A);
    if size(b,1) ~= mA, error('Dimension mismatch'); end
    x0 = x0(:);
    if numel(x0) ~= n, error('x0 must have length %d', n); end

    % Default gamma = 1/L where L = (1/M)*||A||^2 + 2*lambda
    if isempty(gamma)
        try
            s = svds(A,1); normA2 = s^2;
        catch
            normA2 = norm(A)^2;
        end
        L = normA2 + lambda;
        gamma = 1 / max(L, eps);
    end

    % ----------------------
    % Precompute & helpers
    % ----------------------
    AtA = A' * A;
    Atb = A' * b;

    % Gradient of smooth part
    gradf = @(x) (AtA * x - Atb) + lambda * x;

    % Objective (smooth + quadratic) and smooth-only objective (for guard if needed)
    obj = @(x) 0.5* norm(A*x - b)^2 + 0.5*lambda * (x'*x);
    
    % Projection onto nonnegative orthant
    proj_pos = @(v) max(v, 0);

    % ----------------------
    % History preallocation
    % ----------------------
    x_hist = zeros(n, maxit + 2);
    f_hist = zeros(1, maxit + 2);
    t_hist = zeros(1, maxit + 2);
    res_hist = zeros(1, maxit);
    t_start = tic;

    % ----------------------
    % Initialization: initial gradient/prox step
    % ----------------------
    x = x0;
    f0 = obj(x);
    y = x0;                     % candidate variable (y_k)
    g = x - gamma * gradf(x);   % gradient step before prox
    x_new = proj_pos(g);       % proximal (projection) step
    fk = obj(x_new);
    % Circular buffers for Anderson memory (store residuals and candidates)
    Rbuf = zeros(n, m+1);  % residuals r = g - y
    Gbuf = zeros(n, m+1);  % gradient-step candidates g
    buf_count = 0;
    buf_idx = 0;

    % Store first residual
    r = g - y;
    buf_idx = mod(buf_idx, m+1) + 1;
    Rbuf(:, buf_idx) = r;
    Gbuf(:, buf_idx) = g;
    buf_count = min(buf_count + 1, m+1);

    % Record x0 and x1
    x_hist(:,1) = x;          f_hist(1) = f0;   t_hist(1) = toc(t_start);
    x_hist(:,2) = x_new;      f_hist(2) = fk; t_hist(2) = toc(t_start);

    if opts.verbose
        fprintf('Iter %4d: obj = %.6e, res = %.2e\n', 1, f_hist(2), abs(fk - f0)/(1+f0));
    end

    x_k = x_new; y_k = g;  % current iterate and candidate
    k_final = 1;
    k_accept = 0;
    % ----------------------
    % Main loop
    % ----------------------
    for iter = 1:maxit
        % compute gradient-step candidate and residual
        grad = gradf(x_k);
        gk = x_k - gamma * grad;
        rk = gk - y_k;

        % update circular buffers
        buf_idx = mod(buf_idx, m+1) + 1;
        Rbuf(:, buf_idx) = rk;
        Gbuf(:, buf_idx) = gk;
        buf_count = min(buf_count + 1, m+1);

        % extract last buf_count columns in correct time order
        if buf_count == m+1
            idx = mod((buf_idx-(m+1)+1 : buf_idx)-1, m+1) + 1;
        else
            idx = 1:buf_count;
        end
        Rk = Rbuf(:, idx);
        Gk = Gbuf(:, idx);
        s = size(Rk,2);

        % Anderson coefficients: solve (Rk'*Rk + regI) z = e  and normalize
        H = Rk' * Rk;
        H = (H + H')/2;
        H_reg = H + opts.reg_eps * eye(s);
        e = ones(s,1);
        z = H_reg \ e;
        alpha = z / (e' * z);

        % AA candidate and proximal (projection)
        y_check = Gk * alpha;
        x_check = proj_pos(y_check);

        % Guarded acceptance test
        f_check = obj(x_check);
        lhs = f_check;
        rhs = fk - 1e-4*(gamma/2) * (norm(grad)^2);

        if lhs <= rhs
            % accept AA candidate
            x_new = x_check;
            y_new = y_check;
            accepted = true;
            k_accept = k_accept + 1;
            f_new = lhs;
        else
            % fallback to plain proximal-gradient
            y_new = gk;
            x_new = proj_pos(y_new);
            accepted = false;
            f_new = obj(x_new);
        end

        % record
        col = iter + 2;
        x_hist(:, col) = x_new;
        f_hist(col) = f_new;
        t_hist(col) = toc(t_start);
        res = abs(f_new - fk)/(1+abs(fk));
        res_hist(iter) = res;

        if opts.verbose
            fprintf('Iter %4d: obj = %.6e, res = %.2e, accepted = %d\n',...
                iter+1, f_hist(col), res, double(accepted));
        end

        % stopping
        if res_hist(iter) < tol 
            k_final = iter + 1;
            break;
        end

        % prepare next iter
        x_k = x_new;
        y_k = y_new;
        fk = f_new;
        k_final = iter + 1;
    end

    % ----------------------
    % Trim histories and outputs
    % ----------------------
    K = min(k_final + 1, size(x_hist,2));
    x_hist = x_hist(:, 1:K);
    f_hist = f_hist(1:K);
    t_hist = t_hist(1:K);
    res_hist = res_hist(1:max(0, k_final-1));

    % final outputs
    x = x_hist(:, end);
    info.x_hist = x_hist;
    info.f_hist = f_hist;
    info.t_hist = t_hist;
    info.res_hist = res_hist;
    info.k = k_final;

    if opts.verbose
        if ~isempty(res_hist)
            fprintf('Finished at iter %d, final obj = %.6e, final res = %.2e\n',...
                info.k, f_hist(end), res_hist(end));
        else
            fprintf('Finished at iter %d, final obj = %.6e (no residual recorded)\n',...
                info.k, f_hist(end));
        end
    end
    acc = k_accept;
    toc(t_start);
end
