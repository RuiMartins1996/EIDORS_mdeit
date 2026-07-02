function img = inv_solve_absolute_LM_mdeit(inv_model, data)
    
    data = parse_data(data);

    [maxiter, tol, print_diagnostics] = parse_solver_parameters(inv_model);
    
    function [rk,J] = compute_residual_and_jacobian(x,data,img)

        img.elem_data = x;
        
        simulated_data = fwd_solve_1st_order_mdeit(img);
        rk = calc_difference_data_mdeit(simulated_data,data, img.fwd_model);

        if nargout > 1   % Two output arguments
            J = calc_jacobian_mdeit(img);
        end
    end

    function [r, J] = fun(x)
        if nargout > 1
            [r,J] = compute_residual_and_jacobian(x,data,img);
        else
            r = compute_residual_and_jacobian(x,data,img);
        end
    end
    
    img = calc_jacobian_bkgnd(inv_model);
    x0 = img.elem_data;

    if print_diagnostics
        opts = optimoptions("lsqnonlin",...
            'Algorithm','levenberg-marquardt',...
            'SpecifyObjectiveGradient',true,...
            'Display','iter-detailed',...
            'StepTolerance',tol,...
            'MaxIterations',maxiter,...
            'UseParallel',true);
    else
        opts = optimoptions("lsqnonlin",...
            'Algorithm','levenberg-marquardt',...
            'SpecifyObjectiveGradient',true,...
            'Display','iter-detailed',...
            'StepTolerance',tol,...
            'MaxIterations',maxiter,...
            'UseParallel',true);
    end

    [x,~] = lsqnonlin(@fun,x0,[],[],[],[],[],[],[],opts);

    img.elem_data = x;

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
            eidors_msg('inv_solve_absolute_GN_mdeit: Diagnostics printing enabled', 2);
        else
            print_diagnostics = false;
        end
    end

    % If RtR prior is detected, warn the user that the solver may not be appropriate
    if isfield(inv_model, 'RtR_prior')
        warning('inv_solve_absolute_GN_mdeit: RtR prior detected. This solver will ignore the prior.');
    end
end

