function img = inv_solve_absolute_LM_mdeit(inv_model, data)
    % INV_SOLVE_ABSOLUTE_LM_MDEIT  Absolute MDEIT solver via matrix-free LM.
    %
    % Uses the generic LEVENBERG_MARQUARDT solver (see levenberg_marquardt.m)
    % rather than lsqnonlin. This is necessary because MATLAB's lsqnonlin
    % cannot run matrix-free here:
    %   - 'trust-region-reflective' (the only algorithm accepting
    %     JacobianMultiplyFcn) requires #measurements >= #elements, which
    %     practically never holds for MDEIT/EIT absolute imaging.
    %   - 'levenberg-marquardt' has no matrix-free hook at all; it always
    %     computes JAC'*costFun expecting an explicit Jacobian matrix.
    %
    % The residual function below returns the matrix-free Jacobian
    % operator from calc_jacobian_operator_mdeit as its second output;
    % levenberg_marquardt's internal PCG solve then uses only
    % Jop.mtimes / Jop.mtimes_transp, so J and J'*J are never assembled.
    %
    % Spatial regularization: inv_model.RtR_prior and
    % inv_model.hyperparameter are read via calc_RtR_prior/
    % calc_hyperparameter_mdeit (same convention as
    % inv_solve_diff_GN_one_step_mdeit.m and EIDORS's inv_solve_abs_GN_prior)
    % and folded into the LM normal equations as hp^2*RtR, on top of the
    % usual mu*I damping.
    %
    % See also LEVENBERG_MARQUARDT, CALC_JACOBIAN_OPERATOR_MDEIT, CALC_RTR_PRIOR

    data = parse_data(data);

    [maxiter, tol, print_diagnostics] = parse_solver_parameters(inv_model);
    lm_opts = parse_lm_parameters(inv_model);
    lm_opts.maxiter           = maxiter;
    lm_opts.tol               = tol;
    lm_opts.print_diagnostics = print_diagnostics;
    lm_opts.solve_method      = 'pcg';   % J is matrix-free here, so 'direct' is not an option

    img0 = calc_jacobian_bkgnd(inv_model);
    x0 = img0.elem_data;

    % Tikhonov/NOSER-style spatial prior, matching how
    % inv_solve_diff_GN_one_step_mdeit.m and EIDORS's own
    % inv_solve_abs_GN_prior pull these from inv_model. Folded into the
    % LM normal-equations solve alongside mu*I - see levenberg_marquardt.m.
    lm_opts.RtR   = calc_RtR_prior(inv_model);
    lm_opts.hp    = calc_hyperparameter_mdeit(inv_model);
    lm_opts.x_ref = x0;

    fun = @(x) mdeit_residual(x, data, img0);

    [x, info] = levenberg_marquardt(fun, x0, lm_opts);

    if print_diagnostics
        eidors_msg(sprintf(...
            'inv_solve_absolute_LM_mdeit: exitflag=%d after %d iterations, final cost=%.6g', ...
            info.exitflag, info.iterations, info.cost_history(end)), 2);
    end

    img = img0;
    img.elem_data = x;

end

function [r, Jinfo] = mdeit_residual(x, data, img)
    % Residual function passed to levenberg_marquardt. Follows the
    % required nargout convention: only build the (expensive) Jacobian
    % operator when actually asked for it.
    img.elem_data = x;
    simulated_data = fwd_solve_1st_order_mdeit(img);
    % NOTE: argument order matters - calc_difference_data_mdeit(a,b,.) = b-a.
    % levenberg_marquardt's step (dx = -(J'J+muI)^-1*J'*r, x = x+dx) assumes
    % the "model minus target" residual convention, so r must be sim-meas
    % here (not meas-sim), matching its toy self-test convention.
    r = calc_difference_data_mdeit(data, simulated_data, img.fwd_model);

    if nargout > 1
        % Matrix-free operator (no J ever assembled) - see
        % calc_jacobian_operator_mdeit.m
        Jinfo = calc_jacobian_operator_mdeit(img);
    end
end

% This function returns a vector of the measured data
function data = parse_data(data)

    % Check if data is numeric or a data object with field meas
    if isnumeric(data)
        % If data is numeric, return it as is
        return;
    elseif isstruct(data) && isfield(data, 'meas')
        % If data is a struct with field meas, return the meas field
        data = data.meas;
    else
        error('Data must be either a numeric array or a struct with a field "meas".');
    end

end

function [maxiter, tol, print_diagnostics] = parse_solver_parameters(inv_model)
    if ~isfield(inv_model, 'inv_solve_core')
        maxiter = 20; % default value
        tol = 1e-4; % default value
        print_diagnostics = false; % default value
    else
        if isfield(inv_model.inv_solve_core, 'maxiter')
            maxiter = inv_model.inv_solve_core.maxiter;
        else
            maxiter = 20; % default value
        end

        if isfield(inv_model.inv_solve_core, 'tol')
            tol = inv_model.inv_solve_core.tol;
        else
            tol = 1e-4; % default value
        end
    
        if isfield(inv_model.inv_solve_core,'print_diagnostics') && inv_model.inv_solve_core.print_diagnostics
            print_diagnostics = true;
            eidors_msg('inv_solve_absolute_LM_mdeit: Diagnostics printing enabled', 2);
        else
            print_diagnostics = false;
        end
    end
end

function lm_opts = parse_lm_parameters(inv_model)
    % Reads LM-specific knobs (damping schedule, inner PCG settings)
    % from inv_model.inv_solve_core, if present. See levenberg_marquardt.m
    % for the meaning/defaults of each field - only fields explicitly
    % set here are overridden; the rest fall back to
    % levenberg_marquardt's own defaults.
    lm_opts = struct();

    if isfield(inv_model, 'inv_solve_core')
        c = inv_model.inv_solve_core;
        fields = {'mu0','mu_up','mu_down','mu_min','mu_max', ...
                  'max_mu_tries','pcg_tol','pcg_maxit'};
        for k = 1:numel(fields)
            if isfield(c, fields{k})
                lm_opts.(fields{k}) = c.(fields{k});
            end
        end
    end
end