function [outputArg1,outputArg2] = generalized_cross_validation(imdl)

n_inj = numel(imdl.fwd_model.stimulation);

% If it exists!!!!!!!!!
n_sensors = numel(imdl.fwd_model.sensors);


% Compute jacobian matrix
img_bkgnd= calc_jacobian_bkgnd( inv_model );
J = calc_jacobian_mdeit(img_bkgnd);

% Precompute SVD 
[U,S,V] = svd(J,'econ');

fprintf('Time %2.2f\n',toc);
sigma = diag(S);             % singular values
Uy = U' * data;              % coordinates of data in U-basis
m = size(J,1);
n = length(lambda_vector);

V_lambda = zeros(n,1);

for i = 1:n
    lambda = lambda_vector(i);

    gamma = sigma.^2 ./ (sigma.^2 + m*lambda);    % shrinkage factors

    one_minus_gamma = 1 - gamma;

    numerator = (1/m) * sum( (one_minus_gamma .* Uy).^2 );

    denominator = ( (1/m) * sum(one_minus_gamma) )^2;

    V_lambda(i) = numerator / denominator;
end

optimal_id = find(V_lambda == min(V_lambda));



end

