clc;
rng(0, 'twister');   % Fix random seed for reproducibility
% Problem set up
m = 2000; n = 5000;

% Create HIGHLY ill-conditioned matrix A
[U,~] = qr(randn(m));
[V,~] = qr(randn(n));

cond_num = 1e3;   % condition number
s = logspace(0, -log10(cond_num), min(m,n));
S = diag(s);
A = U(:,1:min(m,n)) * S * V(:,1:min(m,n))';
A = A(:,1:n);

% Scale columns differently → numeric instability
A = A .* repmat(logspace(2,-2,n), m, 1);

% Mild correlated features
A(:,2:end) = A(:,2:end) + 0.05*A(:,1);

% Sparse ground truth but noisy
x_true = sprandn(n,1,0.1);
b = sign(A*x_true + 0.5 * randn(m,1));  % heavy noise

% Regularization parameter
lambda_max = norm((A' * b) / (2*m), Inf);   % lambda_max
lambda_frac = 1e-1;                         
lambda = lambda_frac * lambda_max;

fprintf('lambda_max = %.3e, using lambda = %.3e (frac = %.3g)\n', lambda_max, lambda, lambda_frac);

% Initialize
x0 = zeros(n,1);
L = norm(A)^2/(4*m);
gamma = 1 / L;
delta = 1e-4;
maxit = 1000;
tol = 1e-10;

%======================= RUNNING TEST================================
% Run PGA (Vannila Proximal Gradient Algorithm)
fprintf('Running PGA...\n');
[x_pga, info_pga] = pg_logistic_l1(A, b, lambda, x0, gamma, maxit, tol);

%=======================

% Run Accelerated Proximal Gradient (FISTA)
fprintf('Running FISTA...\n')
[x_fis, info_fis] = fista_logistic_l1(A, b, lambda, x0, gamma, maxit, tol);

%=========================

% Run AA-PG (Anderson Acceleration Proximal Gradient Algorithm)
fprintf('Running AA-PG...\n');
% m = 10
fprintf('Running m = 10...\n');
[x_aa10, info_aa10, acc_aa10] = aa_pg_logistic_l1(A, b, lambda, x0, gamma, 10, maxit, tol);
% m = 20
fprintf('Running m = 20...\n');
[x_aa20, info_aa20, acc_aa20] = aa_pg_logistic_l1(A, b, lambda, x0, gamma, 20, maxit, tol);

%===========================

% Run CROP-PG (CROP Proximal Gradient Algorithm)
fprintf('Running CROP-PG...\n');
% m = 10
fprintf('Running m = 10...\n');
[x_crop10, info_crop10, acc_crop10] = crop_pg_logistic_l1(A, b, lambda, x0, gamma, delta, 10, maxit, tol);
% m = 20
fprintf('Running m = 20...\n');
[x_crop20, info_crop20, acc_crop20] = crop_pg_logistic_l1(A, b, lambda, x0, gamma, delta, 20, maxit, tol);

%=============================================

fprintf('Running CROP-PG-Exact...\n');
% m = 10
fprintf('Running m = 10...\n');
[x_crop_exact10, info_crop_exact10, acc_crop_exact10] = crop_pg_logistic_l1_exact(A, b, lambda, x0, gamma, delta, 10, maxit, tol);
% m = 20
fprintf('Running m = 10...\n');
[x_crop_exact20, info_crop_exact20, acc_crop_exact20] = crop_pg_logistic_l1_exact(A, b, lambda, x0, gamma, delta, 20, maxit, tol);

%=============================================

%=============================================
% Plot Objective vs Iteration
figure('Name','Objective vs Iteration','Color','w');
set(gcf, 'Position', [200 200 800 500]);

% Define colors using MATLAB's modern color order (more pleasing & distinct)
colors = lines(8); % built-in MATLAB color palette

semilogy(info_pga.f_hist - min(info_pga.f_hist), 'Color', colors(1,:), 'LineStyle', '-', 'Marker', 'o', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'PG'); 
hold on;

semilogy(info_fis.f_hist - min(info_fis.f_hist), 'Color', colors(2,:), 'LineStyle', '--', 'Marker', 's', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'FISTA'); 

semilogy(info_aa10.f_hist - min(info_aa10.f_hist), 'Color', colors(3,:), 'LineStyle', '-', 'Marker', 'd', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'AA-PG (m = 10)'); 

semilogy(info_aa20.f_hist - min(info_aa20.f_hist), 'Color', colors(4,:), 'LineStyle', '--', 'Marker', '>', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'AA-PG (m = 20)'); 

semilogy(info_crop10.f_hist - min(info_crop10.f_hist), 'Color', colors(5,:), 'LineStyle', '-', 'Marker', 'x', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'CROP-PG-Inexact (m = 10)'); 
semilogy(info_crop20.f_hist - min(info_crop20.f_hist), 'Color', colors(6,:), 'LineStyle', '-', 'Marker', 'p', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'CROP-PG-Inexact (m = 20)'); 

semilogy(info_crop_exact10.f_hist - min(info_crop_exact10.f_hist), 'Color', colors(7,:), 'LineStyle', '-.', 'Marker', '^', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'CROP-PG-Exact (m = 10)'); 

semilogy(info_crop_exact20.f_hist - min(info_crop_exact20.f_hist), 'Color', colors(8,:), 'LineStyle', '-.', 'Marker', 'p', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'CROP-PG-Exact (m = 20)'); 

grid on;
xlabel('Iteration Number', 'Interpreter','latex', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('$\phi(x_k)-\phi^\star$', 'Interpreter','latex', 'FontSize', 20);
title('Objective vs Iteration', 'FontSize', 20, 'FontWeight', 'bold');
legend('show', 'Location', 'northeast', 'FontSize', 15, 'Box', 'off');
set(gca, 'FontSize', 20, 'LineWidth', 1);

%===========================

% Plot Residual vs Iteration
figure('Name','Residual vs Iteration','Color','w');
set(gcf, 'Position', [200 200 800 500]);

% Use MATLAB's built-in pleasant color set
colors = lines(8);

semilogy(info_pga.res_hist, 'Color', colors(1,:), 'LineStyle', '-', 'Marker', 'o', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'PG'); 
hold on;

semilogy(info_fis.res_hist, 'Color', colors(2,:), 'LineStyle', '--', 'Marker', 's', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'FISTA');

semilogy(info_aa10.res_hist, 'Color', colors(3,:), 'LineStyle', '-', 'Marker', 'd', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'AA-PG (m = 10)');

semilogy(info_aa20.res_hist, 'Color', colors(4,:), 'LineStyle', '--', 'Marker', '>', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'AA-PG (m = 20)');


semilogy(info_crop10.res_hist, 'Color', colors(5,:), 'LineStyle', '-', 'Marker', 'x', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'CROP-PG-Inexact (m = 10)');

semilogy(info_crop20.res_hist, 'Color', colors(6,:), 'LineStyle', '-', 'Marker', 'p', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'CROP-PG-Inexact (m = 20)');

semilogy(info_crop_exact10.res_hist, 'Color', colors(7,:), 'LineStyle', '-.', 'Marker', '^', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'CROP-PG-Exact (m = 10)');

semilogy(info_crop_exact20.res_hist, 'Color', colors(8,:), 'LineStyle', '-.', 'Marker', 'p', ...
    'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'CROP-PG-Exact (m = 20)');

grid on;
xlabel('Iteration Number', 'Interpreter','latex', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('$|\phi(x_{k+1})-\phi(x_k)|/(1+|\phi(x_k)|)$', 'Interpreter','latex', 'FontSize', 20);
title('Residual vs Iteration', 'FontSize', 20, 'FontWeight', 'bold');
legend('show', 'Location', 'northeast', 'FontSize', 15, 'Box', 'off');
set(gca, 'FontSize', 20, 'LineWidth', 1);

%=========================

% Plot Objective vs Time
figure('Name','Objective vs Time','Color','w');
set(gcf, 'Position', [200 200 800 500]);

% Use MATLAB's pleasant, colorblind-friendly default color order
colors = lines(8);

% Each curve with distinct line style and marker
semilogy(info_pga.t_hist,  info_pga.f_hist  - min(info_pga.f_hist), ...
    'Color', colors(1,:), 'LineStyle', '-',  'Marker', 'o', 'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'PG'); 
hold on;

semilogy(info_fis.t_hist,  info_fis.f_hist  - min(info_fis.f_hist), ...
    'Color', colors(2,:), 'LineStyle', '--', 'Marker', 's', 'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'FISTA'); 

semilogy(info_aa10.t_hist, info_aa10.f_hist - min(info_aa10.f_hist), ...
    'Color', colors(3,:), 'LineStyle', '-',  'Marker', 'd', 'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'AA-PG (m = 10)'); 

semilogy(info_aa20.t_hist, info_aa20.f_hist - min(info_aa20.f_hist), ...
    'Color', colors(4,:), 'LineStyle', '--', 'Marker', '>', 'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'AA-PG (m = 20)'); 

semilogy(info_crop10.t_hist, info_crop10.f_hist - min(info_crop10.f_hist), ...
    'Color', colors(5,:), 'LineStyle', '-',  'Marker', 'x', 'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'CROP-PG-Inexact (m = 10)'); 

semilogy(info_crop20.t_hist, info_crop20.f_hist - min(info_crop20.f_hist), ...
        'Color', colors(6,:), 'LineStyle', '-', 'Marker', 'p', 'MarkerSize', 5, ...
        'LineWidth', 2.2, 'DisplayName', 'CROP-PG-Inexact (m = 20)');

semilogy(info_crop_exact10.t_hist,  info_crop_exact10.f_hist  - min(info_crop_exact10.f_hist), ...
    'Color', colors(7,:), 'LineStyle', '-.', 'Marker', '^', 'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'CROP-PG-Exact (m = 10)');

semilogy(info_crop_exact20.t_hist,  info_crop_exact20.f_hist  - min(info_crop_exact20.f_hist), ...
    'Color', colors(8,:), 'LineStyle', '-.', 'Marker', 'p', 'MarkerSize', 5, 'LineWidth', 2.2, 'DisplayName', 'CROP-PG-Exact (m = 20)'); 

% Formatting
grid on;
xlabel('Time (s)', 'Interpreter','latex', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('$\phi(x_k) - \phi^\star$', 'Interpreter','latex', 'FontSize', 20);
title('Objective vs Time', 'FontSize', 20, 'FontWeight', 'bold');
legend('show', 'Location', 'northeast', 'FontSize', 15, 'Box', 'off');
set(gca, 'FontSize', 20, 'LineWidth', 1);


