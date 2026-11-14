%% Small local test for the E2006 Student-t implementation (NO download)
% Requires the modular helper files in this folder on the MATLAB path.
clearvars; clc; rng(8,'twister');
N=50; n=30;
A=sprandn(N,n,0.20);
b=randn(N,1);
nu=0.65;
lambda=1e-2;
p=make_student_problem(A,b,nu,lambda);
x=0.2*randn(n,1);

g=p.grad(x);
h=1e-6;
ix=[1 7 14 23 30];
for j=ix
    ej=zeros(n,1); ej(j)=1;
    gfd=(p.f(x+h*ej)-p.f(x-h*ej))/(2*h);
    relerr=abs(g(j)-gfd)/max([1,abs(g(j)),abs(gfd)]);
    fprintf('gradient component %2d: relative error %.3e\n',j,relerr);
    assert(relerr<1e-6,'Gradient finite-difference test failed');
end
fprintf('Gradient finite-difference test passed.\n');

opts.gamma=0.9/(norm(full(A),2)^2/N);
opts.maxit=35;
opts.tol=1e-14;
opts.memory=4;
opts.delta=1e-4;
opts.theta_aa=1e-4;
opts.eps_reg=1e-12;
opts.print_every=0;
opts.csafe=1;
methods={'PG','FISTA','AA-PG','CROP-Inexact','CROP-Exact','AdaPG'};
for j=1:numel(methods)
    opts.method=methods{j};
    [xout,info]=student_solver_optimized(p,zeros(n,1),opts);
    assert(numel(xout)==n && all(isfinite(xout))); 
    assert(numel(info.f_hist)==info.k && numel(info.res_hist)==info.k);
    assert(numel(info.t_hist)==info.k && all(isfinite(info.f_hist)));
    assert(info.k>=1 && info.k<=opts.maxit);
    if info.k>1
        rcheck=abs(diff(info.f_hist))./(1+abs(info.f_hist(1:end-1)));
        assert(norm(info.res_hist(2:end)-rcheck,inf)<1e-12, ...
            'Relative objective-change histories inconsistent.');
    end
    fprintf('%-14s: k=%d, phi=%.12e, last obj-change=%.4e\n',...
        opts.method,info.k,info.final_obj,info.res_hist(end));
end
fprintf('All five solver smoke tests passed.\n');
