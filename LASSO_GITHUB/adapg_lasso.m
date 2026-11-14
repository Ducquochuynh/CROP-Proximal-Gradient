function [x,info]=adapg_lasso(A,b,lambda,x0,gamma,maxit,tol,q,print_every)
% Optimized LASSO baseline, comparable with optimized CROP-PG.
% Objective: 0.5*||A*x-b||^2+lambda*||x||_1.
% info.k counts proximal updates including initial update where applicable.
% info.t_hist is elapsed WALL time (tic/toc), consistent with CROP-PG.
% info.cpu_time is process CPU time (cputime), available for comparison.
% Stopping: relative objective change, unchanged from uploaded method.
% q is the original AdaPG parameter; do not change the stepsize formula.
if nargin<9||isempty(print_every), print_every=0; end
if nargin<8||isempty(q), q=1.5; end
if ~isscalar(q)||q<1||q>2, error('q must lie in [1,2]'); end
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
f_hist=zeros(1,maxit+1); t_hist=zeros(1,maxit+1); res_hist=zeros(1,maxit);
gamma_hist=zeros(1,maxit); ell_hist=zeros(1,maxit); L_hist=zeros(1,maxit);
t0=tic; cpu0=cputime;
% Preserve original AdaPG initialization x^{-1} -> x^0.
prev=x0; gp=grad(prev);
x=prox(prev-gamma*gp,gamma*lambda);
g=grad(x); phi=obj(x); gamma_prev=gamma; gamma_cur=gamma;
f_hist(1)=phi; t_hist(1)=toc(t0);
for k=1:(maxit-1)
    dx=x-prev; dg=g-gp; nd=norm(dx);
    if nd==0, ell=0; locL=0;
    else, ell=real(dx'*dg)/nd^2; locL=norm(dg)/nd; end
    growth=sqrt(1/q+gamma_cur/gamma_prev);
    val=gamma_cur^2*locL^2-(2-q)*gamma_cur*ell+(1-q);
    d2=2*max(val,0);
    if d2==0, curvature=Inf; else, curvature=1/sqrt(d2); end
    gn=gamma_cur*min(growth,curvature);
    xn=prox(x-gn*g,gn*lambda);
    pn=obj(xn); r=abs(pn-phi)/(1+abs(phi));
    f_hist(k+1)=pn; t_hist(k+1)=toc(t0); res_hist(k)=r;
    gamma_hist(k)=gn; ell_hist(k)=ell; L_hist(k)=locL;
    prev=x; gp=g; x=xn; phi=pn; gamma_prev=gamma_cur; gamma_cur=gn;
    if print_every>0 && (mod(k,print_every)==0||k==1), fprintf('AdaPG iter %d: f=%.6e, res=%.2e\n',k,pn,r); end
    if r<tol, break; end
    if k<maxit-1, g=grad(x); end
end
% info.k counts ALL proximal steps, including AdaPG initialization.
if maxit==1, k=0; end
info.k=k+1; info.loop_iterations=k; info.total_prox_steps=k+1;
info.f_hist=f_hist(1:k+1); info.t_hist=t_hist(1:k+1);
info.res_hist=res_hist(1:k); info.gamma_hist=gamma_hist(1:k);
info.ell_hist=ell_hist(1:k); info.L_hist=L_hist(1:k);
info.q=q; info.elapsed_time=toc(t0); info.cpu_time=cputime-cpu0;
info.gradient_mode=gradient_mode; info.final_objective=phi;
if k>0, info.final_residual=res_hist(k);
else, info.final_residual=NaN; end
fprintf('AdaPG: iterations=%d (including initialization), wall=%.4fs, CPU=%.4fs, objective=%.9e\n',info.k,info.elapsed_time,info.cpu_time,phi);
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
