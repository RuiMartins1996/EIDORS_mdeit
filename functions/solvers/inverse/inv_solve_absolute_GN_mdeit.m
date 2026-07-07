function img = inv_solve_absolute_GN_mdeit(inv_model, data)
    % INV_SOLVE_ABSOLUTE_GN_MDEIT  Absolute MDEIT solver via Gauss-Newton
    % with Armijo backtracking line search.
    %
    % Uses the generic GAUSS_NEWTON solver (see gauss_newton.m)

    % The residual function below returns, as its second output, either:
    %   - the matrix-free Jacobian operator from
    %     calc_jacobian_operator_mdeit (gauss_newton's internal PCG solve
    %     then uses only Jop.mtimes / Jop.mtimes_transp, so J and J'*J
    %     are never assembled), or
    %   - the full Jacobian matrix from calc_jacobian_mdeit (gauss_newton
    %     then forms J'*J and solves the normal equations directly),
    % selected by inv_model.inv_solve_core.do_pcg (default true, i.e.
    % matrix-free).
    %
    % GAUSS_NEWTON itself knows nothing about priors - it just minimizes
    % 0.5*||r||^2. The Tikhonov/NOSER-style spatial prior
    % (inv_model.RtR_prior, inv_model.hyperparameter) is instead folded in
    % here by augmenting the residual/Jacobian: appending hp*R*(x-x_ref)
    % to r and hp*R as extra rows to J, since
    %   0.5*||[r; hp*R*(x-x_ref)]||^2 = 0.5*||r||^2 + 0.5*hp^2*(x-x_ref)'*RtR*(x-x_ref)
    % with RtR = R'*R, which reproduces the usual Gauss-Newton normal
    % equations (J'J+hp^2*RtR)dx=-(J'r+hp^2*RtR*(x-x_ref)) once the outer
    % GN step forms J_aug'*J_aug and J_aug'*r_aug.
    %
    % See also GAUSS_NEWTON, CALC_JACOBIAN_OPERATOR_MDEIT,
    % CALC_JACOBIAN_MDEIT, CALC_RTR_PRIOR, CALC_R_PRIOR

    data = parse_data(data);

    [maxiter, tol, print_diagnostics] = parse_solver_parameters(inv_model);
    do_pcg  = parse_do_pcg(inv_model);
    gn_opts = parse_gn_parameters(inv_model);
    gn_opts.maxiter           = maxiter;
    gn_opts.tol               = tol;
    gn_opts.print_diagnostics = print_diagnostics;
    if do_pcg
        gn_opts.solve_method = 'pcg';
    else
        gn_opts.solve_method = 'direct';
    end

    img0 = calc_jacobian_bkgnd(inv_model);
    x0 = img0.elem_data;

    R  = calc_R_prior(inv_model);
    hp = calc_hyperparameter_mdeit(inv_model);

    fun = @(x) mdeit_residual_with_prior(x, data, img0, R, hp, x0, do_pcg);

    [x, info] = gauss_newton(fun, x0, gn_opts);

    if print_diagnostics
        eidors_msg(sprintf(...
            'inv_solve_absolute_GN_mdeit: exitflag=%d after %d iterations, final cost=%.6g', ...
            info.exitflag, info.iterations, info.cost_history(end)), 2);
    end

    img = img0;
    img.elem_data = x;

end

function [r, Jinfo] = mdeit_residual(x, data, img, do_pcg)
    % Residual function passed to gauss_newton. Follows the required
    % nargout convention: only build the (expensive) Jacobian when
    % actually asked for it.
    img.elem_data = x;
    simulated_data = fwd_solve_1st_order_mdeit(img);
    % NOTE: argument order matters - calc_difference_data_mdeit(a,b,.) = b-a.
    % gauss_newton's step (dx = -(J'J)^-1*J'*r, x = x+alpha*dx) assumes
    % the "model minus target" residual convention, so r must be sim-meas
    % here (not meas-sim), matching its toy self-test convention.
    r = calc_difference_data_mdeit(data, simulated_data, img.fwd_model);

    if nargout > 1
        if do_pcg
            % Matrix-free operator (no J ever assembled) - see
            % calc_jacobian_operator_mdeit.m
            Jinfo = calc_jacobian_operator_mdeit(img);
        else
            % Full Jacobian matrix - see calc_jacobian_mdeit.m
            Jinfo = calc_jacobian_mdeit(img);
        end
    end
end

function [r_aug, Jop_aug] = mdeit_residual_with_prior(x, data, img, R, hp, x_ref, do_pcg)
    % Wraps mdeit_residual, appending the Tikhonov/NOSER prior as an
    % extra "measurement" block hp*R*(x-x_ref) on the residual and hp*R
    % as extra rows on the Jacobian, so gauss_newton sees a plain
    % augmented least-squares problem and needs no special prior handling
    % of its own. The augmented Jacobian is returned as a matrix-free
    % operator or a numeric matrix, matching do_pcg / Jinfo's type.
    if nargout > 1
        [r, Jinfo] = mdeit_residual(x, data, img, do_pcg);
    else
        r = mdeit_residual(x, data, img, do_pcg);
    end

    r_aug = [r; hp*(R*(x - x_ref))];

    if nargout > 1
        if do_pcg
            m = numel(r);
            Jop_aug.mtimes        = @(v) [Jinfo.mtimes(v); hp*(R*v)];
            Jop_aug.mtimes_transp = @(w) Jinfo.mtimes_transp(w(1:m)) + hp*(R.'*w(m+1:end));
        else
            Jop_aug = [Jinfo; hp*R];
        end
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
        maxiter = 10; % default value
        tol = 1e-4; % default value
        print_diagnostics = false; % default value
    else
        if isfield(inv_model.inv_solve_core, 'maxiter')
            maxiter = inv_model.inv_solve_core.maxiter;
        else
            maxiter = 10; % default value
        end

        if isfield(inv_model.inv_solve_core, 'tol')
            tol = inv_model.inv_solve_core.tol;
        else
            tol = 1e-4; % default value
        end

        if isfield(inv_model.inv_solve_core,'print_diagnostics') && inv_model.inv_solve_core.print_diagnostics
            print_diagnostics = true;
            eidors_msg('inv_solve_absolute_GN_mdeit: Diagnostics printing enabled', 2);
        else
            print_diagnostics = false;
        end
    end
end

function do_pcg = parse_do_pcg(inv_model)
    % Selects between the matrix-free Jacobian operator (do_pcg = true,
    % default) and the fully-assembled Jacobian matrix (do_pcg = false).
    do_pcg = true; % default value
    if isfield(inv_model, 'inv_solve_core') && ...
            isfield(inv_model.inv_solve_core, 'do_pcg')
        do_pcg = inv_model.inv_solve_core.do_pcg;
    end
end

function gn_opts = parse_gn_parameters(inv_model)
    % Reads GN-specific knobs (line-search schedule, inner PCG settings)
    % from inv_model.inv_solve_core, if present. See gauss_newton.m for
    % the meaning/defaults of each field - only fields explicitly set
    % here are overridden; the rest fall back to gauss_newton's own
    % defaults.
    gn_opts = struct();

    if isfield(inv_model, 'inv_solve_core')
        c = inv_model.inv_solve_core;
        fields = {'alpha0','alpha_bls','beta_bls','alpha_min', ...
                  'max_bls_tries','pcg_tol','pcg_maxit'};
        for k = 1:numel(fields)
            if isfield(c, fields{k})
                gn_opts.(fields{k}) = c.(fields{k});
            end
        end
    end
end
