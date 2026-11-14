function [x,info] = pgmpc_logistic_l1(A,b,lambda,x0,gamma,maxit,tol,L,opts)
% PGM-pc, Algorithm 5.1, Chang et al., JCAM 485 (2026), 117536.
% min_x mean(log(1+exp(-b.*(A*x)))) + lambda*norm(x,1).
% Uses the SAME objective-change termination test and history convention
% as pg_logistic_l1_optimized for comparable figures.
%
% gamma is lambda_0 (initial step), and L is a known *global* upper bound
% on the Lipschitz constant of the logistic gradient.
% Optional opts: alpha=.99, theta0=.99, rho=.7, lambda_hat=1e8,
%                max_linesearch=200, check_tolerance=1e-12.
if nargin < 9 || isempty(opts), opts=struct(); end
if nargin < 8 || isempty(L), L=normest(A)^2/(4*size(A,1)); end
if nargin < 7 || isempty(tol), tol=1e-8; end
if nargin < 6 || isempty(maxit), maxit=1000; end
if nargin < 4 || isempty(x0), x0=zeros(size(A,2),1); end
if nargin < 5 || isempty(gamma), gamma=1/max(L,eps); end
alpha = getoption(opts,'alpha',0.99);
theta = getoption(opts,'theta0',0.99);
shrink = getoption(opts,'rho',0.7);
lambda_hat = getoption(opts,'lambda_hat',1e8);
max_ls = getoption(opts,'max_linesearch',200);
check_tol = getoption(opts,'check_tolerance',1e-12);
[m,n]=size(A); b=b(:); x=x0(:);
assert(numel(b)==m && all(abs(b)==1) && numel(x)==n,'Invalid dimensions/labels');
assert(lambda>=0 && isfinite(lambda) && isscalar(lambda),'Invalid lambda');
assert(L>0 && isfinite(L) && gamma>0 && isfinite(gamma),'Invalid L/gamma');
assert(alpha>0 && alpha<1 && theta>0 && theta<1 && shrink>0 && shrink<1,'Invalid parameters');
assert(lambda_hat>0 && isfinite(lambda_hat) && maxit>=1 && maxit==floor(maxit),'Invalid cap/maxit');
assert(tol>0 && isfinite(tol),'Invalid tolerance');

f_hist=zeros(1,maxit+1); t_hist=zeros(1,maxit+1);
res_hist=zeros(1,maxit); lambda_hist=zeros(1,maxit);
correction_trials=0; backtrack_steps=0; n_grad=0; n_obj=0;
x_prev=[]; grad_prev=[];
clock=tic; cpustart=cputime;
[fcur,gradcur]=evaluate(x); n_grad=n_grad+1; n_obj=n_obj+1;
f_hist(1)=fcur;
rho_prev=sqrt(2); w_prev=1; delta_prev=1;
lambda_prev=gamma; k_final=0;

for k=1:maxit
    if k==1
        step=lambda_prev; % x1 = J(x0,lambda0), Algorithm 5.1 Step 0
    else
        dprev=x-x_prev; gd=gradcur-grad_prev;
        dprev2=dot(dprev,dprev); gd2=dot(gd,gd);
        if dprev2==0, break; end
        if gd2==0
            local_cap=Inf;
        else
            Lk=sqrt(gd2/dprev2);
            % Eq. (41): alpha/(2*Lk*sqrt(1-alpha/(2*L*lambda_prev)))
            den=1-alpha/(2*L*lambda_prev);
            if den<=0
                % Outside the domain of Eq. (41): fail loudly, rather
                % than silently switching to a different algorithm.
                error('PGM-pc:InvalidStep', ...
                    'Eq. (41) requires lambda_(k-1)>alpha/(2L). Check initial gamma and L.');
            end
            local_cap=alpha/(2*Lk*sqrt(den));
        end
        step=min([rho_prev*lambda_prev,local_cap,lambda_hat]);
    end
    if ~(step>0 && isfinite(step)), error('PGM-pc:InvalidStep','Invalid predicted step'); end
    xnew=shrink_l1(x-step*gradcur,step*lambda);
    [fnew,gradnew]=evaluate(xnew); n_grad=n_grad+1; n_obj=n_obj+1;
    w=1;
    if k>1
        d=xnew-x; d2=dot(d,d);
        % Eq. (12) and improved condition (40).
        gdiff=lambda*(sum(abs(xnew))-sum(abs(x)));
        a=(dot(gradcur,-d)-gdiff)/lambda_prev;
        bcoef=dot(gd,-d)-d2/lambda_prev;
        c=alpha*(dprev2+d2)/4;
        lhs=(a-gd2/(2*L*lambda_prev))*step^2 + bcoef*step;
        if lhs>c+check_tol*max([1,abs(lhs),abs(c)])
            w=0; correction_trials=correction_trials+1;
            % Algorithm 3.1: gradient-only test (6), reuse gradnew.
            ls=0;
            while dot(gradnew-gradcur,d)>d2/step + ...
                    check_tol*max([1,abs(d2/step)])
                ls=ls+1;
                if ls>max_ls
                    error('PGM-pc:LineSearch','Gradient line search exceeded max_linesearch');
                end
                step=shrink*step;
                xnew=shrink_l1(x-step*gradcur,step*lambda);
                [fnew,gradnew]=evaluate(xnew); n_grad=n_grad+1; n_obj=n_obj+1;
                d=xnew-x; d2=dot(d,d);
                backtrack_steps=backtrack_steps+1;
            end
        end
    end
    res=abs(fnew-fcur)/(1+abs(fcur));
    x_prev=x; grad_prev=gradcur;
    x=xnew; gradcur=gradnew; fcur=fnew;
    k_final=k; f_hist(k+1)=fcur; res_hist(k)=res;
    lambda_hist(k)=step; t_hist(k+1)=toc(clock);
    if res<tol || (k>1 && d2==0), break; end
    if k>1
        delta=step/lambda_prev;
        theta=max(theta,1/sqrt(1+delta));
        if w_prev==1 && w==0
            rho_prev=sqrt(theta);
        elseif w_prev==0
            rho_prev=sqrt(1+theta*delta_prev/delta);
        else
            rho_prev=sqrt(1+theta*delta);
        end
        delta_prev=delta;
        w_prev=w;
    end
    lambda_prev=step;
end
info.k=k_final;
info.f_hist=f_hist(1:k_final+1);
info.t_hist=t_hist(1:k_final+1);
info.res_hist=res_hist(1:k_final);
info.lambda_hist=lambda_hist(1:k_final);
info.final_objective=fcur;
info.elapsed_time=toc(clock);
info.cpu_time=cputime-cpustart;
info.setup_time=0;
info.converged=k_final>0 && res_hist(k_final)<tol;
info.correction_trials=correction_trials;
info.backtrack_steps=backtrack_steps;
info.gradient_evals=n_grad;
info.objective_evals=n_obj;

    function [F,G]=evaluate(u)
        z=b.*(A*u);
        % stable softplus(-z) and stable sigmoid(-z)
        v=max(-z,0)+log1p(exp(-abs(z)));
        p=zeros(size(z));
        ix=z>=0;
        ez=exp(-z(ix)); p(ix)=ez./(1+ez);
        ez=exp(z(~ix)); p(~ix)=1./(1+ez);
        F=sum(v)/m+lambda*sum(abs(u));
        G=-(A'*(b.*p))/m;
    end
end

function y=shrink_l1(v,t)
y=sign(v).*max(abs(v)-t,0);
end

function val=getoption(s,name,default)
if isfield(s,name) && ~isempty(s.(name)), val=s.(name); else, val=default; end
end
