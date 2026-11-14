function value = logistic_l1_ops(action,A,b,x,lambda)
% Stable l1 logistic utilities; b must be +/-1.
% action = 'grad', 'smooth', 'obj', 'both' (both returns objective only).
m = numel(b);
z = b.*(A*x);
switch lower(action)
 case 'grad'
    % sigmoid(-z), stable for either sign.
    p = exp(-max(z,0))./(1+exp(-abs(z)));
    value = -(A'*(b.*p))/m;
 case 'smooth'
    u = -z;
    value = mean(max(u,0)+log1p(exp(-abs(u))));
 case 'obj'
    u = -z;
    value = mean(max(u,0)+log1p(exp(-abs(u)))) + lambda*sum(abs(x));
 otherwise
    error('Unknown action');
end
end
