function [x,info,acc] = pg_student_l1(problem,x0,opts)
% Wrapper for the PG method in the E2006 Student-t experiment.
% All underlying implementation is in student_solver.m.
    opts.method = 'PG';
    [x,info] = student_solver_optimized(problem,x0,opts);
    acc = info.n_accepted;
end
