function [x_new, cost_new, alpha, dx, exitflag] = armijo_bls(fun, x, p, cost, slope, opts)
% ARMIJO_BLS  Armijo backtracking line search along a descent direction.
%
%   [x_new, cost_new, alpha, dx, exitflag] = armijo_bls(fun, x, p, cost, slope)
%   [x_new, cost_new, alpha, dx, exitflag] = armijo_bls(fun, x, p, cost, slope, opts)
%
% Given a current point x and a descent direction p (slope = g'*p < 0),
% backtracks along p:
%     x_trial = x + alpha*p
% starting from alpha = opts.alpha0 and shrinking alpha *= opts.beta_bls
% until the Armijo sufficient-decrease condition
%     0.5*||fun(x_trial)||^2 <= cost + opts.alpha_bls*alpha*slope
% is satisfied, or the search stalls (opts.alpha_min / opts.max_bls_tries
% exhausted).
%
% This is deliberately problem-agnostic: it only ever calls fun(x) with
% nargout==1 (residual only), never asking for a Jacobian - as used by
% GAUSS_NEWTON's outer loop.
%
% INPUTS
%   fun    : residual function handle, r = fun(x)  (nargout==1 only)
%   x      : current point, n x 1
%   p      : descent direction, n x 1
%   cost   : 0.5*||fun(x)||^2 at the current point (avoids recomputing it)
%   slope  : directional derivative g'*p of the cost along p (must be < 0)
%   opts   : optional struct, all fields optional:
%     alpha0         initial step length                   (default 1.0)
%     alpha_bls      Armijo sufficient-decrease constant     (default 1e-4)
%     beta_bls       backtracking shrink factor              (default 0.5)
%     alpha_min      smallest step length tried              (default 1e-16)
%     max_bls_tries  max step halvings                       (default 50)
%
% OUTPUTS
%   x_new    : accepted point (== x if the search stalled)
%   cost_new : cost at x_new (== cost if the search stalled)
%   alpha    : step length used (or last tried, if stalled)
%   dx       : x_new - x
%   exitflag : 0 = accepted a sufficient-decrease step
%              2 = stalled (no decrease found along p)
%
% See also GAUSS_NEWTON

if nargin < 6
    opts = struct();
end
opts = default_options(opts);

alpha = opts.alpha0;
accepted = false;
exitflag = 0;
n_bls = 0;

x_new = x;
cost_new = cost;
dx = zeros(size(x));

while ~accepted
    x_trial = x + alpha*p;
    r_trial = fun(x_trial);   % nargout == 1: residual only, cheap(er)
    cost_trial = full_cost(r_trial);

    if cost_trial <= cost + opts.alpha_bls*alpha*slope
        accepted = true;
        dx = alpha*p;
        x_new = x_trial;
        cost_new = cost_trial;
    else
        alpha = alpha*opts.beta_bls;
        n_bls = n_bls + 1;
        if alpha < opts.alpha_min || n_bls >= opts.max_bls_tries
            % No sufficient-decrease step found along p; give up the
            % line search and report a stall (x_new is left unchanged).
            accepted = true;
            exitflag = 2;
        end
    end
end

end


%% ==================== OBJECTIVE (RESIDUAL) ===========================
function c = full_cost(r)
% 0.5*||r||^2
c = 0.5*(r.'*r);
end


%% ==================== OPTIONS ========================================
function opts = default_options(opts)

defaults.alpha0        = 1.0;
defaults.alpha_bls     = 1e-4;
defaults.beta_bls      = 0.5;
defaults.alpha_min     = 1e-16;
defaults.max_bls_tries = 50;

fn = fieldnames(defaults);
for k = 1:numel(fn)
    if ~isfield(opts, fn{k})
        opts.(fn{k}) = defaults.(fn{k});
    end
end
end
