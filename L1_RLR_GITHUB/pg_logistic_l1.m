function [x,info]=pg_logistic_l1(A,b,lambda,x0,gamma,maxit,tol)
% Optimized PG for l1 logistic regression.
% Consistent information fields: info.k proximal steps, info.f_hist,
% info.t_hist (elapsed wall seconds), info.res_hist, info.elapsed_time,
% info.cpu_time, info.final_objective. History starts with x0 at index 1.
if nargin < 7 || isempty(tol), tol = 1e-8; end
if nargin < 6 || isempty(maxit), maxit = 1000; end
if nargin < 4 || isempty(x0), x0=zeros(size(A,2),1); end
[mA,n]=size(A); b=b(:);x0=x0(:);
if numel(b)~=mA || numel(x0)~=n || any(abs(b)~=1), error('Dimension or label mismatch'); end
if ~isscalar(lambda) || ~isfinite(lambda) || lambda<0, error('Invalid lambda'); end
if ~isscalar(maxit) || maxit<1 || maxit~=floor(maxit), error('maxit must be integer >=1'); end
if ~isscalar(tol) || ~isfinite(tol) || tol<=0, error('Invalid tol'); end
if nargin < 5 || isempty(gamma), gamma=4*mA/max(normest(A)^2,eps); end
if ~isscalar(gamma) || ~isfinite(gamma) || gamma<=0, error('Invalid gamma'); end
obj=@(u) logistic_l1_ops('obj',A,b,u,lambda);
grad=@(u) logistic_l1_ops('grad',A,b,u,lambda);
smooth=@(u) logistic_l1_ops('smooth',A,b,u,lambda);
prox=@(u) sign(u).*max(abs(u)-gamma*lambda,0);
f_hist=zeros(1,maxit+1);t_hist=zeros(1,maxit+1);res_hist=zeros(1,maxit);
x=x0;phik=obj(x);f_hist(1)=phik;
clock=tic;cpustart=cputime;k_final=0;
for k=1:maxit
    x_new=prox(x-gamma*grad(x));
    phi=obj(x_new); res=abs(phi-phik)/(1+abs(phik));
    x=x_new;phik=phi;k_final=k;
    f_hist(k+1)=phi;res_hist(k)=res;t_hist(k+1)=toc(clock);
    if res<tol, break; end
end
info.k=k_final;
info.f_hist=f_hist(1:k_final+1);
info.t_hist=t_hist(1:k_final+1);
info.res_hist=res_hist(1:k_final);
info.final_objective=info.f_hist(end);
info.elapsed_time=toc(clock);
info.cpu_time=cputime-cpustart;
info.setup_time=0;
info.converged= k_final>0 && info.res_hist(end)<tol;
end
