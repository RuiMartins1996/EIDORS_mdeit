function [x, info] = gauss_newton(fun, x0, opts)
% GAUSS_NEWTON  Generic Gauss-Newton solver with Armijo backtracking line search.
%
%   [x, info] = gauss_newton(fun, x0)
%   [x, info] = gauss_newton(fun, x0, opts)
%
% Solves min_x 0.5*||r(x)||^2 by the plain (undamped) Gauss-Newton method:
% each outer step solves the normal equations (J'*J)*p = -J'*r for the
% full GN step direction p, then backtracks along p with an Armijo
% sufficient-decrease line search - as in EIDORS's
% INV_SOLVE_ABS_GN_CONSTRAIN - rather than damping the normal equations
% the way LEVENBERG_MARQUARDT does.
%
% This is deliberately problem-agnostic AND prior-agnostic: it doesn't
% know about EIT/MDEIT, forward solves, or spatial regularization. If you
% need a Tikhonov/NOSER-style prior, fold it into fun yourself by
% augmenting the residual and Jacobian before returning them, e.g.
%     r_aug          = [r; hp*R*(x-x_ref)]
%     J_aug.mtimes        = @(v) [J*v;         hp*R*v]
%     J_aug.mtimes_transp = @(w) J'*w(1:m) + hp*R'*w(m+1:end)
% (see INV_SOLVE_ABSOLUTE_LM_MDEIT for a worked example of exactly this
% augmentation, there used with LEVENBERG_MARQUARDT). gauss_newton then
% never needs to know a prior exists.
%
% INPUTS
%   fun   : function handle with the calling convention
%               r       = fun(x)          % nargout == 1
%               [r, J]  = fun(x)          % nargout == 2
%           i.e. fun MUST only compute (the expensive) J when called
%           with two output arguments - only compute it on evaluations
%           that need it, not on every trial step during the line search.
%
%           J (the second output) can be EITHER:
%             - a numeric (dense or sparse) matrix, size [m x n], or
%             - a struct with fields
%                   J.mtimes(v)         -> returns J*v   (m x 1)
%                   J.mtimes_transp(w)  -> returns J'*w   (n x 1)
%
%   x0    : initial guess, n x 1 (or reshape-able to a column)
%
%   opts  : optional struct, all fields optional:
%     maxiter        outer GN iterations                  (default 20)
%     tol            ||dx|| convergence tolerance          (default 1e-4)
%     alpha0         initial line-search step length       (default 1.0)
%     alpha_bls      Armijo sufficient-decrease constant    (default 1e-4)
%     beta_bls       backtracking shrink factor             (default 0.5)
%     alpha_min      smallest step length tried             (default 1e-16)
%     max_bls_tries  max step halvings per outer step       (default 50)
%     solve_method   'pcg' (default) or 'direct'
%                       'pcg'    - solves (J'J)p=-g with pcg, using only
%                                  J*v / J'*w matvecs (works for BOTH
%                                  dense and matrix-free J, and never
%                                  forms J'*J).
%                       'direct' - forms A = J.'*J explicitly and solves
%                                  p = -A\g directly. ONLY valid if J is
%                                  a numeric matrix (errors otherwise).
%     pcg_tol        inner PCG tolerance                   (default 1e-6)
%     pcg_maxit      inner PCG max iterations               (default 100)
%     print_diagnostics  print per-iteration info           (default false)
%
% OUTPUTS
%   x    : solution
%   info : struct with fields
%     iterations    : number of outer iterations run
%     exitflag      : 1 = converged (||dx||<tol)
%                     2 = stalled (line search failed to find a decrease)
%                     0 = hit maxiter without converging
%     cost_history  : 1 x iterations, 0.5*||r||^2 at each accepted step
%     alpha_history : 1 x iterations, step length used at each accepted step
%
% EXAMPLE (dense Jacobian):
%   fun = @(x) toy_residual_dense(x, data);
%   function [r,J] = toy_residual_dense(x, data)
%       r = model(x) - data;
%       if nargout > 1
%           J = model_jacobian(x);   % ordinary m x n matrix
%       end
%   end
%
% See also PCG, LEVENBERG_MARQUARDT

if nargin < 3
    opts = struct();
end
opts = default_options(opts);

x = x0(:);

cost_history  = zeros(1, opts.maxiter);
alpha_history = zeros(1, opts.maxiter);
exitflag = 0;
n_recorded = 0;

for iter = 1:opts.maxiter

    % Full evaluation (residual + Jacobian) ONCE per outer iteration,
    % regardless of how many step lengths get tried below.
    [r, J] = fun(x);
    cost = full_cost(r);
    g = apply_jacobian(J, r, 'transp');   % g = J'*r  (gradient of 0.5||r||^2)

    p = solve_normal_equations(J, g, opts);   % full (undamped) GN step

    slope = g.'*p;   % directional derivative of the cost along p
    if slope >= 0
        % (J'J) was singular/indefinite along p (e.g. rank-deficient J
        % with no augmented prior block); fall back to steepest descent
        % so the line search still has a valid descent direction.
        p = -g;
        slope = g.'*p;
    end

    alpha = opts.alpha0;
    accepted = false;
    n_bls = 0;
    dx = zeros(size(x));

    while ~accepted
        x_trial = x + alpha*p;
        r_trial = fun(x_trial);   % nargout == 1: residual only, cheap(er)
        cost_trial = full_cost(r_trial);

        if cost_trial <= cost + opts.alpha_bls*alpha*slope
            accepted = true;
            dx = alpha*p;
            x = x_trial;
            cost = cost_trial;
        else
            alpha = alpha*opts.beta_bls;
            n_bls = n_bls + 1;
            if alpha < opts.alpha_min || n_bls >= opts.max_bls_tries
                % No sufficient-decrease step found along p; give up the
                % line search and report a stall (x is left unchanged).
                accepted = true;
                exitflag = 2;
            end
        end
    end

    n_recorded = n_recorded + 1;
    cost_history(n_recorded)  = cost;
    alpha_history(n_recorded) = alpha;

    if opts.print_diagnostics
        fprintf('GN iter %3d: cost=%.6g  alpha=%.3g  ||dx||=%.3g\n', ...
            iter, cost, alpha, norm(dx));
    end

    if exitflag == 2
        break;
    end
    if norm(dx) < opts.tol
        exitflag = 1;
        break;
    end
end

info.iterations    = iter;
info.exitflag      = exitflag;
info.cost_history  = cost_history(1:n_recorded);
info.alpha_history = alpha_history(1:n_recorded);

end


%% ==================== NORMAL-EQUATIONS SOLVE ========================
function p = solve_normal_equations(J, g, opts)
% Solves (J'*J)*p = -g for p (the undamped Gauss-Newton step).

switch opts.solve_method
    case 'direct'
        if ~isnumeric(J)
            error('gauss_newton:direct_requires_matrix', ...
                ['solve_method = ''direct'' requires J to be a numeric ' ...
                 'matrix; got a matrix-free operator. Use ''pcg'' instead.']);
        end
        A = J.'*J;
        p = -(A\g);

    case 'pcg'
        apply_A = @(v) apply_jacobian(J, apply_jacobian(J, v, 'notransp'), 'transp');
        p = pcg(apply_A, -g, opts.pcg_tol, opts.pcg_maxit);

    otherwise
        error('gauss_newton:bad_solve_method', ...
            'Unknown solve_method ''%s''. Use ''pcg'' or ''direct''.', opts.solve_method);
end
end


%% ==================== OBJECTIVE (RESIDUAL) ===========================
function c = full_cost(r)
% 0.5*||r||^2
c = 0.5*(r.'*r);
end


%% ==================== JACOBIAN DISPATCH ============================
function w = apply_jacobian(J, v, mode)
% Applies J*v ('notransp') or J'*v ('transp'), whether J is a numeric
% matrix or a matrix-free operator struct with .mtimes/.mtimes_transp.

if isnumeric(J)
    switch mode
        case 'notransp', w = J*v;
        case 'transp',   w = J.'*v;
    end
elseif isstruct(J) && isfield(J,'mtimes') && isfield(J,'mtimes_transp')
    switch mode
        case 'notransp', w = J.mtimes(v);
        case 'transp',   w = J.mtimes_transp(v);
    end
else
    error('gauss_newton:bad_jacobian', ...
        ['J must be either a numeric matrix or a struct with fields ' ...
         '''mtimes'' and ''mtimes_transp''.']);
end
end


%% ==================== OPTIONS ========================================
function opts = default_options(opts)

defaults.maxiter           = 20;
defaults.tol               = 1e-4;
defaults.alpha0            = 1.0;
defaults.alpha_bls         = 1e-4;
defaults.beta_bls          = 0.5;
defaults.alpha_min         = 1e-16;
defaults.max_bls_tries     = 50;
defaults.solve_method      = 'pcg';
defaults.pcg_tol           = 1e-6;
defaults.pcg_maxit         = 100;
defaults.print_diagnostics = false;

fn = fieldnames(defaults);
for k = 1:numel(fn)
    if ~isfield(opts, fn{k})
        opts.(fn{k}) = defaults.(fn{k});
    end
end
end


%% ==================== OPTIONAL SELF-TEST ============================
% Uncomment to sanity-check the solver on a toy dense problem, and to
% confirm 'pcg' and 'direct' solve_methods agree (both operate on the
% same dense J, just via different linear solves):
%
%   rng(0);
%   m = 50; n = 10;
%   A_true = randn(m,n);
%   x_true = randn(n,1);
%   data = A_true*x_true + 0.01*randn(m,1);
%
%   function [r,J] = toy_fun(x, A_true, data)
%       r = A_true*x - data;
%       if nargout > 1
%           J = A_true;   % linear problem: Jacobian is constant
%       end
%   end
%
%   x0 = zeros(n,1);
%   opts.solve_method = 'direct';
%   [x_direct, info_direct] = gauss_newton(@(x) toy_fun(x,A_true,data), x0, opts);
%
%   opts.solve_method = 'pcg';
%   [x_pcg, info_pcg] = gauss_newton(@(x) toy_fun(x,A_true,data), x0, opts);
%
%   disp(norm(x_direct - x_pcg) / norm(x_direct));  % should be ~1e-4..1e-6
%   disp(norm(x_direct - x_true) / norm(x_true));   % should be small
