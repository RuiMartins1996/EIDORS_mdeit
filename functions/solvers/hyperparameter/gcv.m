function [hp, imdl] = gcv(imdl)
% GCV: Generalized Cross Validation for hyperparameter selection
%   Computes the optimal regularization parameter using GCV
%   for Tikhonov regularization
%
% [hp, imdl] = gcv(imdl)
%   hp    => optimal hyperparameter
%   imdl  => EIDORS inverse model

% Implementation of "Generalized Cross-Validation as a Method for Choosing
% a Good Ridge Parameter"


% Check if RtR_prior field exists
if ~isfield(imdl, 'RtR_prior')
    % No prior specified, set to Tikhonov with warning
    warning('gcv: No RtR_prior specified. Setting to @prior_tikhonov');
    imdl.RtR_prior = @prior_tikhonov;
end

% Check if the prior is Tikhonov regularization
if isa(imdl.RtR_prior, 'function_handle')
    prior_name = func2str(imdl.RtR_prior);
    if ~strcmp(prior_name, 'prior_tikhonov')
        error('gcv: RtR_prior must be Tikhonov regularization (@prior_tikhonov), but got %s', prior_name);
    end
elseif ischar(imdl.RtR_prior)
    if ~strcmp(imdl.RtR_prior, 'prior_tikhonov')
        error('gcv: RtR_prior must be Tikhonov regularization (prior_tikhonov), but got %s', imdl.RtR_prior);
    end
else
    error('gcv: RtR_prior must be a function handle or string');
end

fprintf('Selecting hyperparameter with GCV ...\n');

% Get background Jacobian
img_bkgnd = calc_jacobian_bkgnd(imdl);
J = calc_jacobian_mdeit(img_bkgnd);

if ~isfield(imdl, 'hyperparameter') || ~isfield(imdl.hyperparameter, 'data')
    error('gcv: Exact GCV requires imdl.hyperparameter.data to contain the data vector');
end
data = imdl.hyperparameter.data;

% Compute the reduced SVD of the Jacobian once up front
[U, S, ~] = svd(J, 'econ');
sigma = diag(S);

if size(data,1) ~= size(J,1)
    error('gcv: Data vector has %d rows but Jacobian has %d measurements', size(data,1), size(J,1));
end
Uy = U' * data;

% Get number of measurements
n_meas = size(J, 1);

% Search bounds for hyperparameter (lambda)
% Search over lambda from 1e-5 to 1e1 (lambda^2 will be used in the solver)

% Use fminsearch to find the lambda that minimizes the GCV criterion
opt = optimset('MaxIter',100, 'TolX', 1e-4);

% Compute the gcv cost function on a lambda vector to visualize the GCV curve
% lambda_vec = logspace(log10(eps(1)), 5, 100);
% gcv_scores = arrayfun(@(lambda) gcv_criterion(lambda, sigma, Uy), lambda_vec);

% Visualize the GCV curve
% figure;
% semilogx(lambda_vec, gcv_scores, 'b-', 'LineWidth', 2);
% xlabel('Regularization Parameter \lambda');
% ylabel('GCV Score');
% title('GCV Curve for Hyperparameter Selection');

% Use fminsearch to find the optimal lambda, and check if it terminates successfully
[lambda_opt, ~, exitflag] = fminsearch(@(lambda) gcv_criterion(lambda, sigma, Uy), 1.0, opt);

switch exitflag
    case 1
        fprintf('GCV optimization converged successfully.\n');
    case 0
        warning('gcv: fminsearch reached maximum iterations without convergence');
    case -1
        warning('gcv: fminsearch was terminated by the output function');
end

% Return optimal hyperparameter (hp^2 = n_meas*lambda_opt)
hp = sqrt(n_meas*lambda_opt);

end

% =========================================================================
function gcv_score = gcv_criterion(lambda, sigma, Uy)
% Compute GCV criterion for a given lambda

n_meas = size(Uy,1);

% Tikhonov shrinkage factors from the reduced SVD of J
gamma = sigma.^2 ./ (sigma.^2 + n_meas*lambda);
one_minus_gamma = 1 - gamma;

% Golub's numerator uses the projected data coefficients
numerator = sum((one_minus_gamma .* Uy).^2, 1) / n_meas;
numerator = mean(numerator);

% Trace term in the denominator
denominator = (sum(one_minus_gamma) / n_meas)^2;

if denominator <= 0
    gcv_score = 1e10;
else
    gcv_score = numerator / denominator;
end

end
