%% E2006-tfidf: nonconvex Student's-t-type L1-regularized robust regression
% Standalone MATLAB R2024b script: ALL functions appear below in this file.
% Real source: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#E2006-tfidf
%
% Comparison: PG, FISTA, AA-PG (m=10/20), CROP-PG-Inexact (m=10/20),
%             CROP-PG-Exact (m=10/20), AdaPG, PGM-pc (experimental).
%
% Original stopping rule:
%        abs(phi(x_k)-phi(x_{k-1})) / (1+abs(phi(x_{k-1}))) < tol.
% Original plot aesthetics preserved; PGM-pc added as tenth method.
%
% Model: phi(x) = nu^2/(2*N) * sum(log(1+((A*x-b)/nu).^2)) + lambda*norm(x,1).

clearvars; clc; close all;
rng(0,'twister');
root = fileparts(mfilename('fullpath'));
if isempty(root), root = pwd; end

%% ========= USER SETTINGS =========
cfg.data_dir = fullfile(root,'data');
cfg.output_dir = fullfile(root,'results');
cfg.max_rows = inf;
cfg.maxit = 2000;
cfg.adapg_q = 1.5;             % AdaPG parameter in [1,2]
cfg.get_test_data = true;
cfg.tol = 1e-13;                 % SAME as uploaded Run_l1_logistic.m
cfg.memories = [10 20];
cfg.delta = 1e-4;
cfg.csafe = 10000;               % Positive c; Inf disables consistency guard
cfg.theta_aa = 1e-4;
cfg.eps_reg = 1e-12;
cfg.lambda_fraction = 1e-2;
cfg.nu_factor = 0.25;            % nu=0.5*std(standardized training labels)=0.5
cfg.standardize_targets = true;  % use training mean/std, shared by methods
cfg.norm_safety = 1.05;
cfg.print_every = 25;           % set 1 to print every iteration
cfg.get_test_data = false;      % TRUE for official 3,308-row test evaluation
cfg.save_figures = true;
cfg.demo_mode = false;          % TRUE: tiny SYNTHETIC code check ONLY
% ==================================

if numel(cfg.memories) ~= 2 || any(cfg.memories < 1)
    error('Exactly two positive memory settings are required.');
end
if ~isfolder(cfg.output_dir), mkdir(cfg.output_dir); end

%% Real LIBSVM sparse data (outside the benchmark clocks)
if cfg.demo_mode
    fprintf('DEMO MODE: synthetic data; DO NOT use these results in the paper.\n');
    N0=100; n0=75;
    A = sprandn(N0,n0,0.1);
    b_orig = A*(sprandn(n0,1,0.1)) + 0.15*randn(N0,1);
    source_file = 'SYNTHETIC DEMO';
else
    fprintf('Loading E2006-tfidf training observations...\n');
    [A,b_orig,source_file] = load_e2006_data(cfg.data_dir,'train',cfg.max_rows);
end
assert(issparse(A),'A must be sparse; never use full(A) on E2006.');
[N,n] = size(A);
if N < 2 || n < 2, error('Need at least two samples and features.'); end
fprintf('Dataset: %d samples, %d features, %d nonzeros.\n',N,n,nnz(A));

% Scale targets using training data only; keeps problem numerically sensible.
label_mean=0; label_std=1;
if cfg.standardize_targets
    label_mean = mean(b_orig);
    label_std = std(b_orig);
    if label_std <= 0 || ~isfinite(label_std), error('Labels have zero/invalid std.'); end
end
b = (b_orig-label_mean)/label_std;
clear b_orig;
nu = cfg.nu_factor*std(b);
if ~isfinite(nu) || nu <= 0, error('Invalid robust-loss scale nu.'); end

% f'(0), lambda, and the spectral norm are precomputed OUTSIDE solver times.
x0 = zeros(n,1);
prob0 = make_student_problem(A,b,nu,0);
lambda_max = norm(prob0.grad(x0),inf);
lambda = cfg.lambda_fraction * lambda_max;
problem = make_student_problem(A,b,nu,lambda);
try
    smax = svds(A,1);
catch ME
    warning('E2006:NormEstimate','svds unsuccessful (%s); using normest.', ME.message);
    smax = normest(A);
end
if ~isfinite(smax) || smax <= 0, error('Unable to estimate ||A||_2.'); end
% Global Lipschitz bound: |rho''(r)| <= 1 for this Student-t-type loss.
L = (cfg.norm_safety*smax)^2/N;
gamma = 1/L;
fprintf('nu=%.5g, lambda=%.6e, smax=%.6e, L=%.6e, gamma=%.6e\n', ...
    nu,lambda,smax,L,gamma);
fprintf('Stopping: |phi_new-phi_old|/(1+|phi_old|) < %.2e\n',cfg.tol);

%% Run the original nine methods plus PGM-pc, same data/stopping rule
labels = {'PG','FISTA','AA-PG (m = 10)','AA-PG (m = 20)', ...
    'CROP-PG-Inexact (m = 10)','CROP-PG-Inexact (m = 20)', ...
    'CROP-PG-Exact (m = 10)','CROP-PG-Exact (m = 20)','AdaPG','PGM-pc'};
% Make the labels reflect any changed memory selection.
labels{3}=sprintf('AA-PG (m = %d)',cfg.memories(1));
labels{4}=sprintf('AA-PG (m = %d)',cfg.memories(2));
labels{5}=sprintf('CROP-PG-Inexact (m = %d)',cfg.memories(1));
labels{6}=sprintf('CROP-PG-Inexact (m = %d)',cfg.memories(2));
labels{7}=sprintf('CROP-PG-Exact (m = %d)',cfg.memories(1));
labels{8}=sprintf('CROP-PG-Exact (m = %d)',cfg.memories(2));
methods = {'PG','FISTA','AA-PG','AA-PG', ...
    'CROP-Inexact','CROP-Inexact','CROP-Exact','CROP-Exact','AdaPG','PGM-pc'};
mem = [0,0,cfg.memories,cfg.memories,cfg.memories,0,0];
nmethods = numel(methods);
infos = cell(nmethods,1);
sols = cell(nmethods,1);
for ii=1:nmethods
    opts.method = methods{ii};
    opts.memory = max(1,mem(ii));
    opts.gamma = gamma;
    opts.maxit = cfg.maxit;
    opts.tol = cfg.tol;
    opts.delta = cfg.delta;
    opts.theta_aa = cfg.theta_aa;
    opts.eps_reg = cfg.eps_reg;
    opts.print_every = cfg.print_every;
    opts.adapg_q = cfg.adapg_q;
    opts.csafe = cfg.csafe;
    opts.L = L; % Full smooth gradient Lipschitz bound; common to all methods.
    fprintf('\n=========== %s ===========\n', labels{ii});
    if strcmp(methods{ii},'PGM-pc')
        [sols{ii}, infos{ii}] = pgmpc_studentt_optimized(problem,x0,opts);
    else
        [sols{ii}, infos{ii}] = student_solver_optimized(problem,x0,opts);
    end
end

%% Optional true holdout-set regression evaluation, excluded from method CPU
TestRMSE = nan(nmethods,1);
TestRobustLoss = nan(nmethods,1);
if cfg.get_test_data && ~cfg.demo_mode
    [A_test,b_test_orig] = load_e2006_data(cfg.data_dir,'test',inf);
    if size(A_test,2) ~= n
        error('Train/test feature dimensions differ: %d vs %d.',n,size(A_test,2));
    end
    b_test = (b_test_orig-label_mean)/label_std;
    for ii=1:nmethods
        r_test = A_test*sols{ii}-b_test;
        TestRMSE(ii) = sqrt(mean((label_std*r_test).^2));
        TestRobustLoss(ii) = nu^2/2 * mean(log1p((r_test/nu).^2));
    end
end

%% One COMMON reference across all algorithms, not one per algorithm.
phi_ref=inf;
for ii=1:nmethods
    phi_ref=min(phi_ref,min(infos{ii}.f_hist));
end
% This is only best OBSERVED value for a NONCONVEX test.
fprintf('\nCommon best-observed objective phi_ref = %.16g\n',phi_ref);

%% Summary table and reproducibility files
Method = string(labels(:));
Iterations = zeros(nmethods,1);
CPUTime_s = zeros(nmethods,1);
WallTime_s = zeros(nmethods,1);
FinalObjective = zeros(nmethods,1);
FinalRelChange = zeros(nmethods,1);
% Common normalized proximal-gradient residual at each final iterate.
% Compute only AFTER solver timers; use gamma=1/L for ALL algorithms,
% including adaptive AdaPG and PGM-pc. Not a stopping condition.
PGResidual = zeros(nmethods,1);
AcceptedPct = nan(nmethods,1);
StopMet = false(nmethods,1);
CorrectionTrials = nan(nmethods,1);
BacktrackSteps = nan(nmethods,1);
GradientEvals = nan(nmethods,1);
for ii=1:nmethods
    v = infos{ii};
    Iterations(ii) = v.k;
    CPUTime_s(ii) = v.cpu_time;
    WallTime_s(ii) = v.elapsed_time;
    FinalObjective(ii) = v.f_hist(end);
    FinalRelChange(ii) = v.res_hist(end);
    xx = sols{ii};
    grad_xx = problem.grad(xx);
    prox_x = problem.prox(xx - gamma*grad_xx, gamma*lambda);
    PGResidual(ii) = norm((xx - prox_x)/gamma) / max(1,norm(xx));
    if ~isfinite(PGResidual(ii))
        error('Nonfinite PG residual for %s.', labels{ii});
    end
    StopMet(ii) = v.stop_met;
    if ii>=3 && ii<=8, AcceptedPct(ii) = 100*v.n_accepted/max(1,v.k-1); end
    if isfield(v,'correction_trials')
        CorrectionTrials(ii)=v.correction_trials;
        BacktrackSteps(ii)=v.backtrack_steps;
        GradientEvals(ii)=v.gradient_evals;
    end
end
summary = table(Method,Iterations,WallTime_s,CPUTime_s,FinalObjective,FinalRelChange,PGResidual, ...
    AcceptedPct,StopMet,TestRMSE,TestRobustLoss, ...
    CorrectionTrials,BacktrackSteps,GradientEvals);
disp(summary);
fprintf('\nFinal normalized PG residuals (common gamma = %.8e):\n',gamma);
disp(summary(:,{'Method','Iterations','CPUTime_s','FinalObjective','PGResidual','StopMet'}));
writetable(summary, fullfile(cfg.output_dir,'E2006_10Methods_summary.csv'));
save(fullfile(cfg.output_dir,'E2006_10Methods_results.mat'), ...
    'infos','sols','summary','labels','phi_ref','cfg','nu','lambda', ...
    'PGResidual','gamma','L','smax','label_mean','label_std','source_file','-v7');

%% Three plots, retaining user's original colors/markers/style/size.
plot_original_three(infos,labels,phi_ref,cfg.output_dir,cfg.save_figures);
fprintf('\nAll results saved under:\n%s\n',cfg.output_dir);
fprintf(['IMPORTANT: This is NONCONVEX. phi_ref is the smallest observed\n', ...
         'objective across all histories, NOT a certified optimum.\n', ...
         'PGM-pc is an experimental extension: convex theorem does not apply.\n']);


%% ======================================================================
%%                         LOCAL MATLAB FUNCTIONS
%% ======================================================================
function p = make_student_problem(A,b,nu,lambda)
    [N,n] = size(A);
    p.N=N; p.n=n; p.lambda=lambda;
    p.f = @(x) (nu^2/(2*N))*sum(log1p(((A*x-b)/nu).^2));
    p.grad = @(x) student_grad(x,A,b,nu,N);
    p.h = @(x) lambda*norm(x,1);
    p.prox = @(v,t) sign(v).*max(abs(v)-t,0);
    p.phi = @(x) p.f(x)+p.h(x);
end

function g = student_grad(x,A,b,nu,N)
    r=A*x-b;
    g=(A'*(r./(1+(r/nu).^2)))/N;
end

function [x,info] = student_solver_optimized(p,x0,opts)
% PG, FISTA, AA-PG, CROP-(in)exact and AdaPG updates are adapted from
% the original logistic L1 files; only f and grad have changed.
% Stopping: relative successive objective change as in original scripts.
    % Consistency guard only acts on CROP candidates; Inf recovers baseline.
    if ~isfield(opts,'csafe') || isempty(opts.csafe), opts.csafe=1; end
    csafe=opts.csafe;
    assert(isscalar(csafe) && isreal(csafe) && (isinf(csafe) || ...
           (isfinite(csafe) && csafe>0)), 'csafe must be positive or Inf');
    if ~isfield(opts,'print_every'),opts.print_every=25;end
    gamma=opts.gamma;
    maxit=opts.maxit;
    tol=opts.tol;
    memory=opts.memory;
    typ=opts.method;
    if maxit<1, error('maxit must be >=1.'); end
    if strcmp(typ,'AdaPG')
        if ~isfield(opts,'adapg_q') || isempty(opts.adapg_q)
            q_adapg=1.5;
        else
            q_adapg=opts.adapg_q;
        end
        assert(isscalar(q_adapg) && isfinite(q_adapg) && ...
            q_adapg>=1 && q_adapg<=2,'AdaPG requires q in [1,2].');
        gamma_previous=gamma;  % gamma_{-1}
        gamma_current=gamma;   % gamma_0, shared first PG step
        gamma_history=zeros(maxit,1);
        gamma_history(1)=gamma_current;
    end

    fh=zeros(maxit,1);
    th=zeros(maxit,1);
    rh=zeros(maxit,1);
    nacc=0;
    reject_objective=0; reject_consistency=0;
    accepted_hist=false(maxit,1);
    consistency_hist=nan(maxit,1);
    bound_hist=nan(maxit,1);
    cpu0=cputime;
    t0=tic;
    x=x0(:);
    phi_old=p.phi(x);
    grad_old=p.grad(x);
    y=x;                  % auxiliary point (for AA/CROP), FISTA momentum
    tf=1;                 % FISTA momentum parameter
    % Fixed-capacity buffers avoid reallocating very wide E2006 vectors.
    Rbuf=[]; Gbuf=[]; aa_count=0;
    Ybuf=[]; Hbuf=[]; crop_count=0;
    if strcmp(typ,'AA-PG')
        Rbuf=zeros(p.n,memory+1); Gbuf=zeros(p.n,memory+1);
    elseif strncmp(typ,'CROP-',5)
        Ybuf=zeros(p.n,memory); Hbuf=zeros(p.n,memory);
    end
    F=zeros(p.n,1);
    stop_met=false;

    % First standard PG step shared by all nine methods.
    % For AdaPG: x0 supplied by caller is x^{-1}; xnew is x^{0}.
    if strcmp(typ,'AdaPG')
        adapg_xprevious=x;
        adapg_gprevious=grad_old;
    end
    z0=x-gamma*grad_old;
    xnew=p.prox(z0,gamma*p.lambda);
    phi_new=p.phi(xnew);
    if ~isfinite(phi_new), error('Nonfinite objective in first step.'); end
    k=1;
    iter_done=1;
    fh(k)=phi_new;
    rh(k)=abs(phi_new-phi_old)/(1+abs(phi_old));
    if strcmp(typ,'AA-PG')
        Rbuf(:,1)=z0-x;
        Gbuf(:,1)=z0;
        aa_count=1;
        y=z0;
    elseif strncmp(typ,'CROP-',5)
        y=z0;
        grad_c=p.grad(xnew); % required for initial CROP residual
        F=xnew-gamma*grad_c-y;
        if strcmp(typ,'CROP-Exact'), grad_old=grad_c; end
    elseif strcmp(typ,'FISTA')
        tfnew=(1+sqrt(1+4*tf^2))/2;
        y=xnew+((tf-1)/tfnew)*(xnew-x);
        tf=tfnew;
    end
    previous_iterate=x;
    x=xnew;
    phi_old=phi_new;
    th(k)=toc(t0);
    stop_met=(rh(k)<tol);
    report_iter(typ,k,phi_new,rh(k),false,opts.print_every);
    % For a common initialization, the first PG step is still subject to
    % the requested stopping rule (unlike an extra uncounted warm-up).

    for k=2:maxit
        if stop_met, break; end
        old_x=x;
        accepted=false;
        pnew=NaN;
        ynew=[];
        Fnew=[];

        switch typ
            case 'AdaPG'
                % Secant curvature at (x^{k-1},x^{k-2}).
                grad_current=p.grad(old_x);
                sk=old_x-adapg_xprevious;
                yk=grad_current-adapg_gprevious;
                sn2=sk'*sk;
                if sn2>0 && isfinite(sn2)
                    ellk=(sk'*yk)/sn2;
                    Lk=norm(yk)/sqrt(sn2);
                else
                    % No informative secant: the curvature bound vanishes.
                    ellk=0;
                    Lk=0;
                end
                curvature=gamma_current^2*Lk^2 ...
                    -(2-q_adapg)*gamma_current*ellk+1-q_adapg;
                grow=sqrt(1/q_adapg+gamma_current/gamma_previous);
                if max(curvature,0)==0
                    bound=inf; % screenshot convention 1/0=Inf
                else
                    bound=1/sqrt(2*max(curvature,0));
                end
                gamma_next=gamma_current*min(grow,bound);
                if ~isfinite(gamma_next) || gamma_next<=0
                    error('AdaPG:InvalidStep','AdaPG stepsize became invalid at iteration %d.',k);
                end
                xnew=p.prox(old_x-gamma_next*grad_current, ...
                    gamma_next*p.lambda);
                adapg_xprevious=old_x;
                adapg_gprevious=grad_current;
                gamma_previous=gamma_current;
                gamma_current=gamma_next;
                gamma_history(k)=gamma_next;

            case 'PG' 
                grad_current=p.grad(old_x);
                zpg=old_x-gamma*grad_current;
                xnew=p.prox(zpg,gamma*p.lambda);

            case 'FISTA'
                gy=p.grad(y);
                xnew=p.prox(y-gamma*gy,gamma*p.lambda);
                tfnew=(1+sqrt(1+4*tf^2))/2;
                ynew=xnew+((tf-1)/tfnew)*(xnew-old_x);
                tf=tfnew;

            case 'AA-PG'
                grad_current=p.grad(old_x);
                zpg=old_x-gamma*grad_current;
                gk=zpg;
                rk=gk-y;
                aa_count=min(aa_count+1,memory+1);
                if aa_count==memory+1 && k>memory+1
                    Rbuf(:,1:memory)=Rbuf(:,2:memory+1);
                    Gbuf(:,1:memory)=Gbuf(:,2:memory+1);
                end
                Rbuf(:,aa_count)=rk;
                Gbuf(:,aa_count)=gk;
                [alpha,ok]=affine_coeff(Rbuf(:,1:aa_count),opts.eps_reg);
                if ok
                    yaa=Gbuf(:,1:aa_count)*alpha;
                    xaa=p.prox(yaa,gamma*p.lambda);
                    f_aa=p.f(xaa);
                    % Keep original AA-PG *smooth-only* acceptance test.
                    if isfinite(f_aa) && ...
                       f_aa <= p.f(old_x)-opts.theta_aa*gamma/2*sum(grad_current.^2)
                        xnew=xaa; ynew=yaa; accepted=true;
                        pnew=f_aa+p.h(xnew); nacc=nacc+1;
                    end
                end
                if ~accepted
                    ynew=zpg;
                    xnew=p.prox(ynew,gamma*p.lambda);
                end

            case {'CROP-Inexact','CROP-Exact'}
                ytilde=y+F;
                xtilde=p.prox(ytilde,gamma*p.lambda);
                Ftilde=xtilde-gamma*p.grad(xtilde)-ytilde;
                crop_count=min(crop_count+1,memory);
                if crop_count==memory && k>memory+1
                    Ybuf(:,1:memory-1)=Ybuf(:,2:memory);
                    Hbuf(:,1:memory-1)=Hbuf(:,2:memory);
                end
                Ybuf(:,crop_count)=y;
                Hbuf(:,crop_count)=F;
                Yk=[Ybuf(:,1:crop_count),ytilde];
                Hk=[Hbuf(:,1:crop_count),Ftilde];
                [alpha,ok]=affine_coeff(Hk,opts.eps_reg);
                if ok
                    ycr=Yk*alpha;
                    xcr=p.prox(ycr,gamma*p.lambda);
                    fcr=p.phi(xcr);
                    % Original safeguard: eps_k=1/(k_iter+1)^2, with
                    % k_iter=k-1 so eps_k=1/k^2 (first trial at k=2).
                    eps_k=1/k^2;
                    objective_ok=isfinite(fcr) && ...
                        fcr <= phi_old+(eps_k-opts.delta)*norm(xcr-old_x)^2;
                    % For the inexact variant, y+F is NOT necessarily z_k.
                    if isinf(csafe)
                        consistency_ok=true;
                        bound_hist(k)=Inf;
                    else
                        if strcmp(typ,'CROP-Exact')
                            z_current=old_x-gamma*grad_old;
                        else
                            z_current=old_x-gamma*p.grad(old_x);
                        end
                        consistency_hist(k)=norm(ycr-z_current);
                        bound_hist(k)=csafe*norm(old_x-previous_iterate);
                        consistency_ok=isfinite(consistency_hist(k)) && ...
                            consistency_hist(k)<=bound_hist(k);
                    end
                    if ~objective_ok, reject_objective=reject_objective+1; end
                    if objective_ok && ~consistency_ok
                        reject_consistency=reject_consistency+1;
                    end
                    if objective_ok && consistency_ok
                        xnew=xcr; ynew=ycr; pnew=fcr;
                        accepted=true; nacc=nacc+1;
                        if strcmp(typ,'CROP-Inexact')
                            Fnew=Hk*alpha;
                        end
                    end
                end
                if ~accepted
                    % CROP-Inexact needs grad(old_x) ONLY on fallback.
                    % CROP-Exact can reuse the exact gradient from F(old_x).
                    if strcmp(typ,'CROP-Exact')
                        grad_current=grad_old;
                    else
                        if ~isinf(csafe) && exist('z_current','var')
                            ynew=z_current;
                        else
                            ynew=old_x-gamma*p.grad(old_x);
                        end
                    end
                    if strcmp(typ,'CROP-Exact')
                        ynew=old_x-gamma*grad_current;
                    end
                    xnew=p.prox(ynew,gamma*p.lambda);
                end

            otherwise
                error('Unknown method: %s',typ);
        end

        if ~isfinite(pnew), pnew=p.phi(xnew); end
        if ~isfinite(pnew) || any(~isfinite(xnew))
            error('%s produced a nonfinite iterate/objective at k=%d.',typ,k);
        end
        if strncmp(typ,'CROP-',5)
            if strcmp(typ,'CROP-Exact')
                % Exact residual variant always evaluates true F(xnew).
                grad_old=p.grad(xnew);
                Fnew=xnew-gamma*grad_old-ynew;
            elseif isempty(Fnew)
                % Inexact residual needs an exact residual only on fallback.
                Fnew=xnew-gamma*p.grad(xnew)-ynew;
            end
            F=Fnew;
        end
        if ~isempty(ynew), y=ynew; end
        accepted_hist(k)=accepted;
        rh(k)=abs(pnew-phi_old)/(1+abs(phi_old));
        fh(k)=pnew;
        th(k)=toc(t0);
        iter_done=k;
        previous_iterate=old_x;
        x=xnew;
        phi_old=pnew;
        stop_met=(rh(k)<tol);
        report_iter(typ,k,pnew,rh(k),accepted,opts.print_every);
    end

    info.elapsed_time=toc(t0);
    info.cpu_time=cputime-cpu0;
    info.c=csafe;
    info.accept_hist=accepted_hist(1:iter_done);
    info.accept_rate=nacc/max(1,iter_done-1);
    info.consistency_hist=consistency_hist(1:iter_done);
    info.bound_hist=bound_hist(1:iter_done);
    info.reject_objective=reject_objective;
    info.reject_consistency=reject_consistency;
    info.f_hist=fh(1:iter_done);
    info.t_hist=th(1:iter_done);
    info.res_hist=rh(1:iter_done);
    info.k=iter_done;
    info.final_obj=fh(iter_done);
    info.stop_met=stop_met;
    info.n_accepted=nacc;
    if strcmp(typ,'AdaPG')
        info.gamma_hist=gamma_history(1:iter_done);
        info.adapg_q=q_adapg;
    end
end


function [alpha,ok]=affine_coeff(R,eps_reg)
    % Regularized normal equations, as in original AA/CROP implementations.
    p=size(R,2);
    H=R'*R;
    H=(H+H')/2;
    u=(H+eps_reg*eye(p))\ones(p,1);
    d=ones(1,p)*u;
    ok=isfinite(d) && d>0 && all(isfinite(u));
    if ok
        alpha=u/d;
        ok=all(isfinite(alpha));
    else
        alpha=[];
    end
end

function report_iter(name,k,f,res,accepted,print_every)
    if print_every>0 && (k==1 || mod(k,print_every)==0 || res<1e-14)
        fprintf('%-14s Iter %4d: obj = %.9e, res = %.2e, accepted = %d\n', ...
            name,k,f,res,accepted);
    end
end

function [A,y,source_file] = load_e2006_data(data_dir,partition,max_rows)
    if nargin<3, max_rows=inf; end
    if ~isfolder(data_dir), mkdir(data_dir); end
    if strcmp(partition,'train')
        basename='E2006.train';
    elseif strcmp(partition,'test')
        basename='E2006.test';
    else
        error('Partition must be train or test.');
    end
    source_file=fullfile(data_dir,basename);
    archive=[source_file '.bz2'];
    url=['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/' ...
         'datasets/regression/' basename '.bz2'];
    if ~isfile(source_file)
        if ~isfile(archive)
            fprintf('Downloading %s from LIBSVM (train archive approx. 180 MB)...\n',basename);
            try
                websave(archive, url, weboptions('Timeout',1200));
            catch ME
                error('E2006:DownloadFailed', ...
                    ['Dataset download failed: %s\nPlease download manually: %s\n', ...
                     'and place it in %s'],ME.message,url,data_dir);
            end
        end
        fprintf('Decompressing %s using macOS bzip2...\n',archive);
        temp=[source_file '.part'];
        if isfile(temp), delete(temp); end
        command=sprintf('bzip2 -dc %s > %s',quote_shell(archive),quote_shell(temp));
        [status,msg]=system(command);
        if status~=0 || ~isfile(temp)
            if isfile(temp), delete(temp); end
            error('E2006:DecompressionFailed', ...
                'macOS bzip2 decompression failed: %s. Archive: %s',msg,archive);
        end
        d=dir(temp);
        if isempty(d) || d.bytes==0
            delete(temp);
            error('E2006:EmptyData','Decompression created an empty file.');
        end
        [success,message]=movefile(temp,source_file,'f');
        if ~success, error('E2006:MoveFailed','%s',message); end
    end

    f=fopen(source_file,'r');
    if f<0, error('E2006:ReadFailed','Cannot read %s',source_file); end
    cleaner=onCleanup(@()fclose(f)); %#ok<NASGU>
    n=150360;  % official E2006-tfidf feature dimension
    cap=1000000;
    I=zeros(cap,1,'uint32');
    J=zeros(cap,1,'uint32');
    V=zeros(cap,1);
    y=zeros(20000,1);
    nnz_done=0; rows=0;
    fprintf('Parsing %s into a MATLAB sparse matrix (max_rows=%g)...\n', ...
        basename,max_rows);
    while rows<max_rows
        line=fgetl(f);
        if ~ischar(line), break; end
        if isempty(strtrim(line)), continue; end
        fields=sscanf(strrep(line,':',' '),'%f');
        if isempty(fields), continue; end
        if mod(numel(fields)-1,2)~=0
            error('E2006:BadFormat','Malformed LIBSVM line %d.',rows+1);
        end
        rows=rows+1;
        if rows>numel(y), y(end+10000,1)=0; end
        y(rows)=fields(1);
        cols=fields(2:2:end);
        vals=fields(3:2:end);
        if any(cols<1) || any(cols>n) || any(cols~=floor(cols))
            error('E2006:BadIndex','Bad feature index at line %d.',rows);
        end
        count=numel(cols);
        while nnz_done+count>cap
            grow=max(cap,1000000);
            I(end+grow,1)=uint32(0);
            J(end+grow,1)=uint32(0);
            V(end+grow,1)=0;
            cap=cap+grow;
        end
        idx=nnz_done+(1:count);
        I(idx)=uint32(rows);
        J(idx)=uint32(cols);
        V(idx)=vals;
        nnz_done=nnz_done+count;
        if mod(rows,1000)==0
            fprintf('  parsed %d rows, %d nonzero feature entries\n',rows,nnz_done);
        end
    end
    if rows==0, error('E2006:NoData','No observations parsed from %s.',source_file); end
    y=y(1:rows);
    % MATLAB sparse expects numeric index vectors; conversions are local.
    A=sparse(double(I(1:nnz_done)),double(J(1:nnz_done)),V(1:nnz_done),rows,n);
    fprintf('Loaded %s: %d x %d, nnz=%d.\n',basename,rows,n,nnz(A));
end

function q=quote_shell(path)
    % POSIX-shell safe quotation for paths with spaces and apostrophes.
    apostrophe=char(39);
    seq=[apostrophe '"' apostrophe '"' apostrophe];
    q=[apostrophe strrep(char(path),apostrophe,seq) apostrophe];
end

function plot_original_three(infos,labels,phi_ref,output_dir,save_figures)
    colors=lines(numel(infos));
    styles={'-','--','-','--','-','-','-.','-.','--','-'};
    markers={'o','s','d','>','x','p','^','p','h','v'};
    fig_pos=[200 200 800 500];
    floor_gap=eps(max(1,abs(phi_ref)));  % only for log visualization
    for fig_id=1:3
        switch fig_id
            case 1
                figtitle='Objective vs Iteration';
                x_label='Iteration Number';
                y_label='$\phi(x_k)-\phi_{\mathrm{ref}}$';
                outname='E2006_10Methods_Objective_vs_Iteration';
            case 2
                figtitle='Residual vs Iteration';
                x_label='Iteration Number';
                y_label='$|\phi(x_{k+1})-\phi(x_k)|/(1+|\phi(x_k)|)$';
                outname='E2006_10Methods_Residual_vs_Iteration';
            case 3
                figtitle='Objective vs Time';
                x_label='Time (s)';
                y_label='$\phi(x_k)-\phi_{\mathrm{ref}}$';
                outname='E2006_10Methods_Objective_vs_Time';
        end
        f=figure('Name',figtitle,'Color','w');
        set(f,'Position',fig_pos);
        hold on;
        for j=1:numel(infos)
            infj=infos{j};
            if fig_id==1
                xp=1:numel(infj.f_hist);
                yp=infj.f_hist-phi_ref;
                yp(~isfinite(yp) | yp<=floor_gap)=NaN;
            elseif fig_id==2
                xp=1:numel(infj.res_hist);
                yp=infj.res_hist;
                yp(~isfinite(yp) | yp<=0)=NaN;
            else
                xp=infj.t_hist;
                yp=infj.f_hist-phi_ref;
                yp(~isfinite(yp) | yp<=floor_gap)=NaN;
            end
            semilogy(xp,yp,'Color',colors(j,:),'LineStyle',styles{j}, ...
                'Marker',markers{j},'MarkerSize',5,'LineWidth',2.2, ...
                'DisplayName',labels{j});
        end
        grid on;
        xlabel(x_label,'Interpreter','latex','FontSize',20,'FontWeight','bold');
        ylabel(y_label,'Interpreter','latex','FontSize',20);
        title(figtitle,'FontSize',20,'FontWeight','bold');
        legend('show','Location','northeast','FontSize',15,'Box','off');
        set(gca,'FontSize',20,'LineWidth',1);
        if save_figures
            exportgraphics(f,fullfile(output_dir,[outname '.png']),'Resolution',200);
            exportgraphics(f,fullfile(output_dir,[outname '.pdf']),'ContentType','vector');
        end
    end
end


%% Algorithm 5.1 extension (NONCONVEX Student-t: no convex guarantee)
function [x,info] = pgmpc_studentt_optimized(p,x0,opts)
% PGM-pc (known-L), Algorithm 5.1, Chang et al., JCAM 485 (2026) 117536.
% Student-t robust loss plus l1 regularization. NONCONVEX experiment.
% The convex convergence theorem for Algorithm 5.1 does NOT apply here.
% Objective: p.phi(x) = p.f(x)+p.h(x), p.prox(v,t)=soft(v,t).
% Same output/history convention as student_solver_optimized:
% index 1 records the FIRST proximal update, not x0.
% opts.L must be a known upper bound for Lipschitz constant of p.grad.
% opts.gamma is lambda_0 (initial gradient step).
% No large matrices, no new toolboxes, and no extra objective calls for LS.

if ~isfield(opts,'L') || isempty(opts.L), error('PGM-pc requires opts.L'); end
L=opts.L; lam_prev=opts.gamma; maxit=opts.maxit; tol=opts.tol;
alpha=option_student_pc(opts,'alpha',0.99);
theta=option_student_pc(opts,'theta0',0.99);
shrink=option_student_pc(opts,'rho',0.7);
lam_hat=option_student_pc(opts,'lambda_hat',1e8);
max_ls=option_student_pc(opts,'max_linesearch',100);
check_tol=option_student_pc(opts,'check_tolerance',1e-13);
print_every=option_student_pc(opts,'print_every',0);
assert(L>0 && isfinite(L) && lam_prev>0 && isfinite(lam_prev));
assert(alpha>0 && alpha<1 && theta>0 && theta<1 && shrink>0 && shrink<1);
assert(lam_hat>0 && isfinite(lam_hat) && maxit>=1 && maxit==floor(maxit));
assert(tol>=0 && isfinite(tol) && max_ls>=1);
assert(lam_prev>alpha/(2*L), ...
    'Algorithm 5.1 needs initial lambda0 > alpha/(2*L).');

wall=tic; cpu0=cputime;
fh=zeros(maxit,1);rh=zeros(maxit,1);th=zeros(maxit,1);
lamhist=zeros(maxit,1);
corrections=0; backtracks=0; gradients=0; objectives=0;
xold=x0(:);
phiold=objective(xold);
gold=gradient(xold);
% Step 0: x_1 = prox_{lambda0*g}(x_0-lambda0*grad f(x_0)).
x=p.prox(xold-lam_prev*gold,lam_prev*p.lambda);
phi=objective(x);
kdone=1;
fh(1)=phi;rh(1)=abs(phi-phiold)/(1+abs(phiold));
lamhist(1)=lam_prev;th(1)=toc(wall);
stop_met=rh(1)<tol;
if print_every>0, fprintf('PGM-pc: iter=1, obj=%.12e, relchange=%.3e\n',phi,rh(1)); end
rho_prev=sqrt(2); delta_prev=1; w_prev=1;
% Lazy gradient at x_1: no need if initial relative-change test stops.
if ~stop_met && maxit>1
    g=gradient(x);
else
    g=[];
end
for k=2:maxit
    if stop_met || isequal(x,xold), break; end
    dp=x-xold; gp=g-gold;
    dp2=dot(dp,dp); gp2=dot(gp,gp);
    if dp2==0,break;end
    Lk=sqrt(gp2/dp2);
    denominator=1-alpha/(2*L*lam_prev);
    if denominator<=0, error('PGM-pc invalid prediction denominator'); end
    if Lk==0
        lamcur=min(rho_prev*lam_prev,lam_hat);
    else
        lamcur=min([rho_prev*lam_prev,alpha/(2*Lk*sqrt(denominator)),lam_hat]);
    end
    if ~isfinite(lamcur) || lamcur<=0, error('PGM-pc invalid predicted stepsize'); end
    xnext=p.prox(x-lamcur*g,lamcur*p.lambda);
    d=xnext-x;d2=dot(d,d);
    % Eq. (12) and improved correction condition (40).
    gdifference=p.h(xnext)-p.h(x);
    a=(dot(g,-d)-gdifference)/lam_prev;
    bb=dot(gp,-d)-d2/lam_prev;
    c=alpha*(dp2+d2)/4;
    lhs=(a-gp2/(2*L*lam_prev))*lamcur^2+bb*lamcur;
    w=1;
    if lhs>c+check_tol*max([1,abs(lhs),abs(c)])
        w=0; corrections=corrections+1;
        % Gradient-only linesearch (Algorithm 3.1, Eq. (6)); the
        % correction criterion itself does NOT need grad at xnext.
        gn=gradient(xnext);
        for j=1:max_ls
            if dot(gn-g,d)<=d2/lamcur+check_tol*max(1,abs(d2/lamcur)),break;end
            lamcur=shrink*lamcur;
            xnext=p.prox(x-lamcur*g,lamcur*p.lambda);
            d=xnext-x;d2=dot(d,d);
            gn=gradient(xnext);
            backtracks=backtracks+1;
        end
        if dot(gn-g,d)>d2/lamcur+check_tol*max(1,abs(d2/lamcur))
            error('PGM-pc linesearch failed after %d reductions.',max_ls);
        end
    else
        gn=[]; % computed only when needed for next outer iteration
    end
    phinext=objective(xnext);
    res=abs(phinext-phi)/(1+abs(phi));
    delta=lamcur/lam_prev;
    theta=max(theta,1/sqrt(1+delta));
    if w_prev==1 && w==0
        rho_prev=sqrt(theta);
    elseif w_prev==0
        rho_prev=sqrt(1+theta*delta_prev/delta);
    else
        rho_prev=sqrt(1+theta*delta);
    end
    delta_prev=delta;w_prev=w;lam_prev=lamcur;
    xold=x;gold=g;x=xnext;phi=phinext;
    kdone=k; fh(k)=phi;rh(k)=res;lamhist(k)=lamcur;th(k)=toc(wall);
    stop_met=(res<tol);
    if print_every>0 && (mod(k,print_every)==0 || k==maxit)
        fprintf('PGM-pc: iter=%d, obj=%.12e, relchange=%.3e\n',k,phi,res);
    end
    if ~stop_met && k<maxit
        if isempty(gn),g=gradient(x);else,g=gn;end
    end
end
info.k=kdone;info.f_hist=fh(1:kdone);info.res_hist=rh(1:kdone);
info.t_hist=th(1:kdone); info.lambda_hist=lamhist(1:kdone);
info.final_obj=phi;info.final_objective=phi;info.stop_met=stop_met;
info.elapsed_time=toc(wall);info.cpu_time=cputime-cpu0;
info.setup_time=0;info.iteration_axis=(1:kdone)';
info.n_accepted=0;info.accept_rate=NaN;
info.correction_trials=corrections;info.backtrack_steps=backtracks;
info.gradient_evals=gradients;info.objective_evals=objectives;
info.known_L=L;
    function val=objective(u)
        val=p.phi(u);objectives=objectives+1;
        if ~isfinite(val),error('PGM-pc nonfinite objective');end
    end
    function val=gradient(u)
        val=p.grad(u);gradients=gradients+1;
    end
end
function v=option_student_pc(s,k,d)
if isfield(s,k) && ~isempty(s.(k)),v=s.(k);else,v=d;end
end

