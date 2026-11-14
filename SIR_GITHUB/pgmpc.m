function [x,info] = pgmpc_restoration_optimized(p,x0,opts)
% PGM-pc (known-L), Algorithm 5.1, Chang et al., JCAM 485 (2026) 117536.
% Objective: p.phi(x) = p.f(x)+p.h(x), with p.prox(z,t) = prox_{t*||.||_1}(z).
% Same output/history convention as restoration_solver_with_AdaPG_optimized:
% index 1 records the FIRST proximal update, not x0.
% opts.L must be a known upper bound for Lipschitz constant of p.grad.
% opts.gamma is lambda_0 (initial gradient step).
% No large matrices, no new toolboxes, and no extra objective calls for LS.

if ~isfield(opts,'L') || isempty(opts.L), error('PGM-pc requires opts.L'); end
L=opts.L; lam_prev=opts.gamma; maxit=opts.maxit; tol=opts.tol;
alpha=option(opts,'alpha',0.99);
theta=option(opts,'theta0',0.99);
shrink=option(opts,'rho',0.7);
lam_hat=option(opts,'lambda_hat',1e8);
max_ls=option(opts,'max_linesearch',100);
check_tol=option(opts,'check_tolerance',1e-13);
print_every=option(opts,'print_every',0);
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
function v=option(s,k,d)
if isfield(s,k) && ~isempty(s.(k)),v=s.(k);else,v=d;end
end
