function [grad, obj, backend] = nrr_operators(A,b,lambda)
% Shared operators for all optimized NRR solvers.
% Matrix-free for large/wide systems; Gram matrix for smaller systems.
[rows,n] = size(A);
if n <= 1200 && n*n <= 3*max(nnz(A),1)
    G = A'*A; d = A'*b;
    grad = @(x) G*x-d+lambda*x;
    backend = 'gram';
else
    grad = @(x) A'*(A*x-b)+lambda*x;
    backend = 'matrix-free';
end
obj = @(x) 0.5*sum((A*x-b).^2)+0.5*lambda*sum(x.^2);
end
