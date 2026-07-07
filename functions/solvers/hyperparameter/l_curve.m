function lambda_opt = l_curve(inv_model)

fprintf('Selecting hyperparameter with L-curve method ...\n');

lambda_vec = parse_hyperparameter_data(inv_model);

% Allocate for residual and solution norms
r_norm = zeros(1,length(lambda_vec));
x_norm = zeros(1,length(lambda_vec));

% Fetch dm from inv_model
if ~isfield(inv_model, 'hyperparameter') || ~isfield(inv_model.hyperparameter, 'data')
    error('l_curve: inv_model.hyperparameter.data must contain the data vector');
end
dm  = inv_model.hyperparameter.data;

% Get background Jacobian
img_bkgnd = calc_jacobian_bkgnd(inv_model);
J = calc_jacobian_mdeit(img_bkgnd);

% Do SVD of J so the linear system solution can be computed efficiently for each lambda
[U, S, V] = svd(J, 'econ');
s = diag(S);
y = U' * dm;

% Prior
RtR = calc_RtR_prior( inv_model );

for i = 1:length(lambda_vec) 

    hp = lambda_vec(i);    
    
    % Use SVD to compute the solution for each hyperparameter (lambda)  
    %dsigma = inv(J.'*J +  hp^2*RtR)*J.'*dm
    A = diag(s.^2) + hp^2 * (V' * RtR * V);    
    z = left_divide(A ,(s .* y));
    dsigma = V * z;

    %dsigma = left_divide((JtJ +  hp^2*RtR),J') * dm;
    
    r_norm(i) = norm(J*dsigma-dm,2);
    x_norm(i) = norm(dsigma,2);
end

% plot(log10(r_norm),log10(x_norm))

%% Compute the curvature by fitting a smoothing cubic spline to graph

% Given data
r = r_norm(:);
xnorm = x_norm(:);

% Sort by residual norm (important for monotone x)
[rs, idx] = sort(r);
xs = xnorm(idx);

% Log-log scale for L-curve
lr = log(rs);
lx = log(xs);

% --- Use monotone-preserving piecewise cubic interpolation ---
lr_dense = linspace(min(lr), max(lr), 200);
lx_dense = interp1(lr, lx, lr_dense, 'pchip');

% --- Numerical derivatives using finite differences ---
d1 = gradient(lx_dense, lr_dense);          % first derivative
d2 = gradient(d1, lr_dense);               % second derivative

% --- Curvature formula: κ = |y''| / (1 + y'^2)^(3/2)
kappa = abs(d2) ./ (1 + d1.^2).^(3/2);

% --- Interpolate curvature back to original lr points if needed
kappa_interp = interp1(lr_dense, kappa, lr, 'pchip');

% Find maximum curvature (corner of L-curve)
[~, imax] = max(kappa_interp);
%opt_lr = lr_dense(imax);
%opt_lx = lx_dense(imax);

% If the maximum curvature is at the boundary, warn the user
if imax == 1 || imax == length(kappa_interp)
    warning('l_curve: Maximum curvature occurs at the boundary of the L-curve. Consider adjusting the lambda_vec range.');
end

% Assuming lambda_vec corresponds to lr_dense
lambda_opt = lambda_vec(imax);

% Convert back to linear scale
% opt_r = exp(opt_lr);
% opt_x = exp(opt_lx);

% fprintf('Optimal residual norm = %.4e\n', opt_r);
% fprintf('Optimal solution norm = %.4e\n', opt_x);
% fprintf('Maximum curvature = %.4e\n', kappa_interp(imax));
% fprintf('Optimal hyperparameter = %.4e\n', lambda_opt);


end


function lambda_vec = parse_hyperparameter_data(inv_model)
    
    if ~isfield(inv_model, 'hyperparameter') || ~isfield(inv_model.hyperparameter, 'data')
        error('l_curve: inv_model.hyperparameter.data must contain the data vector');
    end

    % Check if inv_model.hyperparameter.func is l_curve
    if ~isfield(inv_model.hyperparameter, 'func') 
        error('l_curve: inv_model.hyperparameter.func must be set to ''l_curve''');
        % Check if inv_model.hyperparameter.func is @l_curve or 'l_curve'
    elseif ~strcmp(inv_model.hyperparameter.func, 'l_curve') && ~isequal(inv_model.hyperparameter.func, @l_curve)
        error('l_curve: inv_model.hyperparameter.func must be set to ''l_curve'' or @l_curve');
    end 

    % Check if user has supplied a custom lambda_vec
    if isfield(inv_model.hyperparameter, 'lambda_vec') && ~isempty(inv_model.hyperparameter.lambda_vec)
        
        % Check if it is a numeric vector
        if ~isvector(inv_model.hyperparameter.lambda_vec) || ~isnumeric(inv_model.hyperparameter.lambda_vec)
            error('l_curve: inv_model.hyperparameter.lambda_vec must be a numeric vector');
        end

        lambda_vec = inv_model.hyperparameter.lambda_vec;
    else
        % Generate default lambda vector
        lambda_vec = logspace(log10(sqrt(10*eps(1))), 2, 20);
    end

end

