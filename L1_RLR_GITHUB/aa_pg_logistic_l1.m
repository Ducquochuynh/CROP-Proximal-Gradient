function [x,info,acc]=aa_pg_logistic_l1(A,b,lambda,x0,gamma,m,maxit,tol,opts)
% AA-PG preserving original smooth-only acceptance safeguard.
if nargin<9 || isempty(opts),opts=struct();end
if ~isfield(opts,'reg_eps'),opts.reg_eps=1e-12;end
if nargin<8 || isempty(tol),tol=1e-8;end
if nargin<7 || isempty(maxit),maxit=1000;end
if nargin<6 || isempty(m),m=5;end
if nargin<5 || isempty(gamma),gamma=[];end
if m<1 || m~=floor(m),error('m must be positive integer');end
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
x_hist=zeros(n,maxit+1);x_hist(:,1)=x;
Rbuf=zeros(n,m+1);Gbuf=zeros(n,m+1);count=0;pos=0;acc=0;
gradk=grad(x);g0=x-gamma*gradk;x1=prox(g0);
% Original AA residual initialization: g0 - x0
pos=1;Rbuf(:,pos)=g0-x;Gbuf(:,pos)=g0;count=1;
yk=g0;x=x1;phi=obj(x);k_final=1;
x_hist(:,2)=x;f_hist(2)=phi;t_hist(2)=toc(clock);
res_hist(1)=abs(phi-phik)/(1+abs(phik));phik=phi;
accept_hist=false(1,maxit);
for k=2:maxit
    gradk=grad(x);gk=x-gamma*gradk;rk=gk-yk;
    pos=mod(pos,m+1)+1;Rbuf(:,pos)=rk;Gbuf(:,pos)=gk;count=min(count+1,m+1);
    ids=mod((pos-count+1:pos)-1,m+1)+1;
    R=Rbuf(:,ids);G=Gbuf(:,ids);s=size(R,2);
    C=R'*R;C=(C+C')/2;v=(C+opts.reg_eps*eye(s))\ones(s,1);
    if all(isfinite(v)) && isfinite(sum(v)) && abs(sum(v))>eps
        alpha=v/sum(v);yc=G*alpha;xc=prox(yc);
        fcheck=smooth(xc);pass=isfinite(fcheck) && fcheck<=smooth(x)-1e-4*(gamma/2)*(gradk'*gradk);
    else,pass=false;end
    if pass, xnew=xc; ynew=yc;acc=acc+1;accept_hist(k)=true;
    else,ynew=gk;xnew=prox(gk);end
    phi=obj(xnew);res=abs(phi-phik)/(1+abs(phik));
    x=xnew;yk=ynew;phik=phi;k_final=k;
    x_hist(:,k+1)=x;f_hist(k+1)=phi;res_hist(k)=res;t_hist(k+1)=toc(clock);
    if res<tol,break;end
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
info.x_hist=x_hist(:,1:k_final+1);info.accept_hist=accept_hist(1:k_final);
info.accept_rate=acc/max(1,k_final-1);
end
