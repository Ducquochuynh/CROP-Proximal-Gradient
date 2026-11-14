function [x,info]=pg_lasso(A,b,lambda,x0,gamma,maxit,tol,print_every)
% Optimized LASSO baseline, comparable with optimized CROP-PG.
% Objective: 0.5*||A*x-b||^2+lambda*||x||_1.
% info.k counts proximal updates including initial update where applicable.
% info.t_hist is elapsed WALL time (tic/toc), consistent with CROP-PG.
% info.cpu_time is process CPU time (cputime), available for comparison.
% Stopping: relative objective change, unchanged from uploaded method.
if nargin<8||isempty(print_every), print_every=0; end
if nargin < 7 || isempty(tol), tol=1e-8; end
if nargin < 6 || isempty(maxit), maxit=1000; end
[nobs,n]=size(A); b=b(:);
if numel(b)~=nobs, error('Invalid b dimension'); end
if nargin<4 || isempty(x0), x0=zeros(n,1); end
x0=x0(:);
if numel(x0)~=n, error('Invalid x0 dimension'); end
if nargin<5 || isempty(gamma), gamma=1/(svds(A,1)^2); end
if ~isscalar(maxit)||maxit<1||maxit~=floor(maxit), error('maxit must be a positive integer'); end
if ~isscalar(gamma)||~isfinite(gamma)||gamma<=0, error('gamma must be positive'); end
if ~isscalar(tol)||tol<0||~isfinite(tol), error('Invalid tol'); end
if lambda<0, error('lambda must be nonnegative'); end
[grad,obj,gradient_mode]=local_setup(A,b,lambda);
prox=@(v,t) sign(v).*max(abs(v)-t,0);
x=x0; phi=obj(x);
f_hist=zeros(1,maxit+1); t_hist=zeros(1,maxit+1); res_hist=zeros(1,maxit);
f_hist(1)=phi;
t0=tic; cpu0=cputime;
for k=1:maxit
    xn=prox(x-gamma*grad(x),gamma*lambda);
    pn=obj(xn); r=abs(pn-phi)/(1+abs(phi));
    f_hist(k+1)=pn; t_hist(k+1)=toc(t0); res_hist(k)=r;
    x=xn; phi=pn;
    if print_every>0 && (mod(k,print_every)==0||k==1), fprintf('PG iter %d: f=%.6e, res=%.2e\n',k,pn,r); end
    if r<tol, break; end
end
info.f_hist=f_hist(1:k+1); info.t_hist=t_hist(1:k+1);
info.res_hist=res_hist(1:k); info.k=k; info.elapsed_time=toc(t0);
info.cpu_time=cputime-cpu0; info.gradient_mode=gradient_mode;
info.final_objective=phi; info.final_residual=res_hist(k);
fprintf('PG: iterations=%d, wall=%.4fs, CPU=%.4fs, objective=%.9e\n',info.k,info.elapsed_time,info.cpu_time,phi);
end

function [grad,obj,mode] = local_setup(A,b,lambda)
[nobs,n]=size(A);
if n<=min(nobs,1200)
    G=A'*A; g=A'*b;
    grad=@(v) G*v-g;
    mode='Gram';
else
    grad=@(v) A'*(A*v-b);
    mode='matrix-free';
end
obj=@(v) 0.5*sum(abs(A*v-b).^2)+lambda*sum(abs(v));
end
