function img = inv_solve_absolute_GN_mdeit(inv_model, data)

    data = parse_data(data);

    [maxiter, tol, print_diagnostics] = parse_solver_parameters(inv_model);

    % First guess for the conductivity distribution
    img = calc_jacobian_bkgnd(inv_model);

    % Compute the Jacobian and residual for the first iteration
    Jk = calc_jacobian_mdeit(img);
    rk = calc_residual(img, data);

    % Compute objective and gradient
    obj = 0.5*(rk.'*rk);
    gk = Jk.'*rk;

    % Exit immediately if the initial guess is already good enough
    if norm(gk) < tol
        eidors_msg('inv_solve_absolute_GN_mdeit: Converged based on initial gradient norm', 2);
        return;
    end

    for i = 1:maxiter
        % Assemble the normal equations
        A = Jk.'*Jk;
        b = Jk.'*rk;

        % Solve for the update step
        dsigma = left_divide(A,b);
        
        % Update the image
        img.elem_data = img.elem_data + dsigma;

        % Update residual, the Jacobian, objective, and gradient for the next iteration
        Jk = calc_jacobian_mdeit(img);
        rk = calc_residual(img, data);
        obj1 = 0.5*(rk.'*rk);
        gk = Jk.'*rk;

        if print_diagnostics
            report_diagonostics(i, obj1, gk);
        end

        % Check for convergence in the gradient
        if norm(gk) < tol
            eidors_msg('inv_solve_absolute_GN_mdeit: Converged based on gradient norm', 2);
            break;
        end
        % Check for convergence in the objective function
        if abs(obj1 - obj) < tol
            eidors_msg('inv_solve_absolute_GN_mdeit: Converged based on objective function change', 2);
            break;
        end
        obj = obj1;
    end

    % If maximum iterations reached without convergence, issue a warning
    if i == maxiter 
        warning('inv_solve_absolute_GN_mdeit: Maximum iterations reached without convergence');
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

function residual = calc_residual(img, m)
    
    % Simulate the data based on the current image estimate
    simulated_data = fwd_solve_1st_order_mdeit(img);
    
    % Calculate the residual [s.Bx(:);s.By(:);s.Bz(:)]-data(:);
    residual = calc_difference_data_mdeit(simulated_data,m, img.fwd_model);
end

function report_diagonostics(iteration, obj, gk)
    fprintf('Iteration %d: Objective = %.6f, Gradient Norm = %.6f\n', iteration, obj, norm(gk));
end
