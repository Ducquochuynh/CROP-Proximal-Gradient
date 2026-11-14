function g = student_grad(x,A,b,nu,N)
    r=A*x-b;
    g=(A'*(r./(1+(r/nu).^2)))/N;
end
