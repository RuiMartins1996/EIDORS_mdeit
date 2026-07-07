function [x, info] = levenberg_marquardt(fun, x0, opts)
% LEVENBERG_MARQUARDT  Generic damped Gauss-Newton / LM solver.
%
%   [x, info] = levenberg_marquardt(fun, x0)
%   [x, info] = levenberg_marquardt(fun, x0, opts)
%
% Solves min_x 0.5*||r(x)||^2 by Levenberg-Marquardt, WITHOUT any line
% search (per your request) - just classic multiplicative damping:
% shrink mu on an accepted step, grow mu and retry on a rejected one.
%
% This is deliberately problem-agnostic: it doesn't know about EIT,
% forward solves, or anything mdeit-specific. You give it a residual
% function fun; it handles the LM bookkeeping and the damped normal
% equations solve, and works the same whether your Jacobian is an
% ordinary matrix or a matrix-free operator.
%
% INPUTS
%   fun   : function handle with the calling convention
%               r       = fun(x)          % nargout == 1
%               [r, J]  = fun(x)          % nargout == 2
%           i.e. fun MUST only compute (the expensive) J when called
%           with two output arguments - see EXAMPLES below. This mirrors
%           the nested-function pattern used elsewhere (only compute
%           Jacobian on iterations that need it, not on every trial
%           step during the mu search).
%
%           J (the second output) can be EITHER:
%             - a numeric (dense or sparse) matrix, size [m x n], or
%             - a struct with fields
%                   J.mtimes(v)         -> returns J*v   (m x 1)
%                   J.mtimes_transp(w)  -> returns J'*w   (n x 1)
%               (this is exactly the struct calc_jacobian_operator_mdeit
%               returns - pass it straight through as J with no
%               modification needed).
%
%   x0    : initial guess, n x 1 (or reshape-able to a column)
%
%   opts  : optional struct, all fields optional:
%     maxiter        outer LM iterations                (default 20)
%     tol            ||dx|| convergence tolerance        (default 1e-4)
%     mu0            initial damping parameter           (default 1e-3)
%     mu_up, mu_down multiplicative damping adjustment    (default 10,10)
%     mu_min, mu_max damping parameter bounds             (default 1e-12,1e12)
%     max_mu_tries   max damping increases per outer step (default 20)
%     solve_method   'pcg' (default) or 'direct'
%                       'pcg'    - solves (J'J+mu*I)dx=-J'r with pcg,
%                                  using only J*v / J'*w matvecs (works
%                                  for BOTH dense and matrix-free J, and
%                                  never forms J'*J).
%                       'direct' - forms A = J.'*J + mu*I explicitly and
%                                  solves dx = -A\g directly. ONLY valid
%                                  if J is a numeric matrix (errors
%                                  otherwise). Useful as a ground-truth
%                                  comparison against 'pcg' on small
%                                  dense test problems.
%     pcg_tol        inner PCG tolerance                 (default 1e-6)
%     pcg_maxit      inner PCG max iterations             (default 100)
%     print_diagnostics  print per-iteration info         (default false)
%
% OUTPUTS
%   x    : solution
%   info : struct with fields
%     iterations   : number of outer iterations run
%     exitflag     : 1 = converged (||dx||<tol)
%                    2 = stalled (mu search exhausted without improving)
%                    0 = hit maxiter without converging
%     cost_history : 1 x iterations, 0.5*||r||^2 at each accepted step
%     mu_history   : 1 x iterations, mu used at each accepted step
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
% EXAMPLE (matrix-free Jacobian, e.g. MDEIT):
%   function [r,Jinfo] = mdeit_residual(x, data, img)
%       img.elem_data = x;
%       sim = fwd_solve_1st_order_mdeit(img);
%       r = calc_difference_data_mdeit(data, sim, img.fwd_model);  % = sim-data
%       if nargout > 1
%           Jinfo = calc_jacobian_operator_mdeit(img);  % struct, not a matrix
%       end
%   end
%   [x, info] = levenberg_marquardt(@(x) mdeit_residual(x,data,img), x0, opts);
%
% See also PCG, CALC_JACOBIAN_OPERATOR_MDEIT

if nargin < 3
    opts = struct();
end
opts = default_options(opts);

x = x0(:);

r = fun(x);
cost = compute_cost(r);

mu = opts.mu0;

cost_history = zeros(1, opts.maxiter);
mu_history   = zeros(1, opts.maxiter);
exitflag = 0;
n_recorded = 0;

for iter = 1:opts.maxiter

    % Full evaluation (residual + Jacobian) ONCE per outer iteration,
    % regardless of how many mu values get tried below.
    [r, J] = fun(x);
    cost = compute_cost(r);
    g = apply_jacobian(J, r, 'transp');   % g = J'*r

    accepted = false;
    n_mu_tries = 0;
    dx = zeros(size(x));

    while ~accepted
        dx = solve_damped_normal_equations(J, g, mu, opts);

        x_trial = x + dx;
        r_trial = fun(x_trial);   % nargout == 1: residual only, cheap(er)
        cost_trial = compute_cost(r_trial);

        if cost_trial < cost
            accepted = true;
            x = x_trial;
            cost = cost_trial;
            mu = max(mu/opts.mu_down, opts.mu_min);
        else
            mu = min(mu*opts.mu_up, opts.mu_max);
            n_mu_tries = n_mu_tries + 1;
            if mu >= opts.mu_max || n_mu_tries >= opts.max_mu_tries
                % No improving step found from this point; give up the
                % damping search and report a stall.
                accepted = true;
                exitflag = 2;
            end
        end
    end

    n_recorded = n_recorded + 1;
    cost_history(n_recorded) = cost;
    mu_history(n_recorded)   = mu;

    if opts.print_diagnostics
        fprintf('LM iter %3d: cost=%.6g  mu=%.3g  ||dx||=%.3g\n', ...
            iter, cost, mu, norm(dx));
    end

    if exitflag == 2
        break;
    end
    if norm(dx) < opts.tol
        exitflag = 1;
        break;
    end
end

info.iterations   = iter;
info.exitflag     = exitflag;
info.cost_history = cost_history(1:n_recorded);
info.mu_history    = mu_history(1:n_recorded);

end


%% ==================== NORMAL-EQUATIONS SOLVE ========================
function dx = solve_damped_normal_equations(J, g, mu, opts)
% Solves (J'*J + mu*I)*dx = -g for dx.

switch opts.solve_method
    case 'direct'
        if ~isnumeric(J)
            error('levenberg_marquardt:direct_requires_matrix', ...
                ['solve_method = ''direct'' requires J to be a numeric ' ...
                 'matrix; got a matrix-free operator. Use ''pcg'' instead.']);
        end
        n = size(J,2);
        A = J.'*J + mu*speye(n);
        dx = -(A\g);

    case 'pcg'
        apply_A = @(v) apply_jacobian(J, apply_jacobian(J, v, 'notransp'), 'transp') + mu*v;
        dx = pcg(apply_A, -g, opts.pcg_tol, opts.pcg_maxit);

    otherwise
        error('levenberg_marquardt:bad_solve_method', ...
            'Unknown solve_method ''%s''. Use ''pcg'' or ''direct''.', opts.solve_method);
end
end


%% ==================== OBJECTIVE (RESIDUAL) ===========================
function c = compute_cost(r)
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
    error('levenberg_marquardt:bad_jacobian', ...
        ['J must be either a numeric matrix or a struct with fields ' ...
         '''mtimes'' and ''mtimes_transp''.']);
end
end


%% ==================== OPTIONS ========================================
function opts = default_options(opts)

defaults.maxiter          = 10;
defaults.tol              = 1e-4;
defaults.mu0              = 1e-3;
defaults.mu_up            = 10;
defaults.mu_down          = 10;
defaults.mu_min           = 1e-12;
defaults.mu_max           = 1e12;
defaults.max_mu_tries     = 20;
defaults.solve_method     = 'pcg';
defaults.pcg_tol          = 1e-6;
defaults.pcg_maxit        = 100;
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
% same dense J, just via different linear solves - useful before
% trusting the matrix-free path on MDEIT):
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
%   [x_direct, info_direct] = levenberg_marquardt(@(x) toy_fun(x,A_true,data), x0, opts);
%
%   opts.solve_method = 'pcg';
%   [x_pcg, info_pcg] = levenberg_marquardt(@(x) toy_fun(x,A_true,data), x0, opts);
%
%   disp(norm(x_direct - x_pcg) / norm(x_direct));  % should be ~1e-4..1e-6
%   disp(norm(x_direct - x_true) / norm(x_true));   % should be small