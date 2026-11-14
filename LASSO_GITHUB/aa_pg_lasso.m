function [x,info,acc]=aa_pg_lasso(A,b,lambda,x0,gamma,m,maxit,tol,opts)
% Optimized LASSO baseline, comparable with optimized CROP-PG.
% Objective: 0.5*||A*x-b||^2+lambda*||x||_1.
% info.k counts proximal updates including initial update where applicable.
% info.t_hist is elapsed WALL time (tic/toc), consistent with CROP-PG.
% info.cpu_time is process CPU time (cputime), available for comparison.
% Stopping: relative objective change, unchanged from uploaded method.
% Keep original AA acceptance test based on smooth objective only.
if nargin<9||isempty(opts), opts=struct(); end
if ~isfield(opts,'verbose'), opts.verbose=false; end
if ~isfield(opts,'print_every'), opts.print_every=0; end
if ~isfield(opts,'reg_eps'), opts.reg_eps=1e-12; end
if nargin<8||isempty(tol), tol=1e-8; end
if nargin<7||isempty(maxit), maxit=1000; end
if nargin<6||isempty(m), m=5; end
if ~isscalar(m)||m<1||m~=floor(m), error('m must be a positive integer'); end
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
smooth=@(v) 0.5*sum(abs(A*v-b).^2);
t0=tic; cpu0=cputime;
x=x0; y=x0; phi0=obj(x);
y1=x-gamma*grad(x); xk=prox(y1,gamma*lambda);
phik=obj(xk); fk=smooth(xk); yk=y1;
f_hist=zeros(1,maxit+1); t_hist=zeros(1,maxit+1);
res_hist=zeros(1,max(0,maxit-1)); accept_hist=false(1,max(0,maxit-1));
f_hist(1:2)=[phi0,phik]; t_hist(2)=toc(t0);
% First residual/gradient-trial pair: preserve supplied AA history.
R=zeros(n,m+1); G=zeros(n,m+1); count=1; slot=1;
R(:,1)=y1-x0; G(:,1)=y1;
acc=0;
for j=1:(maxit-1)
    gk=grad(xk); zk=xk-gamma*gk; rk=zk-yk;
    slot=mod(slot,m+1)+1;
    R(:,slot)=rk; G(:,slot)=zk; count=min(count+1,m+1);
    % Correct chronological ordering even when buffer wraps.
    ix=mod((slot-count+1:slot)-1,m+1)+1;
    Rk=R(:,ix); Gk=G(:,ix);
    B=Rk'*Rk; B=(B+B')/2; s=size(B,1);
    w=(B+opts.reg_eps*eye(s))\ones(s,1);
    alpha=w/sum(w);
    yc=Gk*alpha; xc=prox(yc,gamma*lambda);
    fc=smooth(xc);
    passed=isfinite(fc)&&fc<=fk-1e-4*(gamma/2)*sum(abs(gk).^2);
    if passed
        xn=xc; yn=yc; fn=fc; acc=acc+1;
    else
        yn=zk; xn=prox(yn,gamma*lambda); fn=smooth(xn);
    end
    pn=fn+lambda*sum(abs(xn)); rr=abs(pn-phik)/(1+abs(phik));
    f_hist(j+2)=pn; t_hist(j+2)=toc(t0); res_hist(j)=rr;
    accept_hist(j)=passed;
    xk=xn; yk=yn; fk=fn; phik=pn;
    if opts.verbose&&(opts.print_every<=1||mod(j+1,opts.print_every)==0)
        fprintf('AA-PG iter %d: f=%.6e, res=%.2e, accepted=%d\n',j+1,pn,rr,passed);
    end
    if rr<tol, break; end
end
if maxit==1, j=0; end
x=xk; k=j+1;
info.k=k; info.f_hist=f_hist(1:k+1); info.t_hist=t_hist(1:k+1);
info.res_hist=res_hist(1:j); info.accept_hist=accept_hist(1:j);
info.accept_rate=acc/max(1,j); info.elapsed_time=toc(t0);
info.cpu_time=cputime-cpu0; info.gradient_mode=gradient_mode;
info.final_objective=phik;
if j>0, info.final_residual=res_hist(j);
else, info.final_residual=abs(phik-phi0)/(1+abs(phi0)); end
fprintf('AA-PG: iterations=%d, wall=%.4fs, CPU=%.4fs, accepted=%d/%d, objective=%.9e\n',k,info.elapsed_time,info.cpu_time,acc,j,phik);
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
