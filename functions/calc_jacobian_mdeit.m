function  J = calc_jacobian_mdeit(img)

% Detect model dimension
%dim = size(img.fwd_model.nodes,2);

recon_mode = check_recon_mode(img);

switch recon_mode
    case 'mdeit1'
        J = calc_jacobian_1axis(img);

    case 'mdeit3'
        [Jx,Jy,Jz] = calc_jacobian_3axis(img);
        J = [Jx;Jy;Jz];
end

end

% RIGHT NOW, calc_jacobian_1axis is using LDL and calc_jacobian_3axis is
% using pcg. Add this as an option!!!!!!!!!!!!!!!!!!!!!

function J = calc_jacobian_1axis(img)

mu0 = img.fwd_model.mu0;

n_nodes = size(img.fwd_model.nodes,1);
n_elem = size(img.fwd_model.elems,1);

num_stim = numel(img.fwd_model.stimulation);
num_sensors = numel(img.fwd_model.sensors);
num_electrodes = numel(img.fwd_model.electrode);

% Compute Gamma matrices
img = compute_gamma_matrices(img);

R = img.fwd_model.R;
G = img.fwd_model.G;

Gamma = img.Gamma;

% Factorize lhs system matrix
A_matrix = lhs_eit_full(img);
F = factorise_symmetric(A_matrix);

% Compute EIT forward solution for each current injection pattern
I = zeros(num_electrodes,num_stim);
for j = 1:num_stim
    I(:,j) = img.fwd_model.stimulation(j).stim_pattern;
end

rhs = sparse(n_nodes+num_electrodes,num_stim);
rhs(end-num_electrodes+1:end,:) = I;

u = solve_fact_multiple_rhs(F,rhs);
u = u(1:end-num_electrodes,:);

% Solve the adjoint problem for each sensor to get lambda vectors
GammaT = Gamma.';

rhs = sparse(n_nodes+num_electrodes,num_sensors);
rhs(1:end-num_electrodes,:) = -GammaT;
lambda = solve_fact_multiple_rhs(F,rhs);
lambda = lambda(1:end-num_electrodes,:);

Gx_times_lambda = G.Gx*lambda;
Gy_times_lambda = G.Gy*lambda;
Gz_times_lambda = G.Gz*lambda;

Gx_times_u = G.Gx*u;
Gy_times_u = G.Gy*u;
Gz_times_u = G.Gz*u;

mu_factor = mu0/(4*pi);

elemV = img.fwd_model.elem_volume(:);      % [numElems × 1]

% We want to broadcast arrays into num_sensors*num_stim*num_elems, in that order, so we avoid a permute in dfd
% Expand elem_volume to cover stim × sensor
elemV = reshape(elemV, [1 1 n_elem]);

GxL = reshape(Gx_times_lambda.', [num_sensors 1 n_elem]); % [: × 1 × numSensors]
GyL = reshape(Gy_times_lambda.', [num_sensors 1 n_elem]);
GzL = reshape(Gz_times_lambda.', [num_sensors 1 n_elem]);

% Expand u-terms to 3D
GxU = reshape(Gx_times_u.', [1 num_stim n_elem]); % [: × numStim × 1]
GyU = reshape(Gy_times_u.', [1 num_stim n_elem]);
GzU = reshape(Gz_times_u.', [1 num_stim n_elem]);

% Compute all dfdx for all sensors+stim
dfdx = elemV .* ( ...
    GxL.*GxU + ...
    GyL.*GyU + ...
    GzL.*GzU );

% Compute all dfdp (also 3D)
Rx_ = reshape(R.Rx, [num_sensors 1 n_elem]);
Ry_ = reshape(R.Ry, [num_sensors 1 n_elem]);
Rz_ = reshape(R.Rz, [num_sensors 1 n_elem]);

% These are the derivatives with respect to sigma of the C components of the Gamma matrix,
dCxdp = ( -Rz_.*GyU + Ry_.*GzU );
dCydp = ( -Rx_.*GzU + Rz_.*GxU );
dCzdp = ( -Ry_.*GxU + Rx_.*GyU );

% The g matrix does not depend on sigma.
g = zeros(num_sensors, 3);
for m = 1:numel(img.fwd_model.sensors)
    g(m,:) = img.fwd_model.sensors(m).axes.axis;
end

dfdp  = mu_factor * (g(:,1).*dCxdp + g(:,2).*dCydp  + g(:,3).*dCzdp);


dfd = dfdx + dfdp;   % size: [numStim × numSensors × numElems]

% Now reshape to match J(ids,:)

% collapse first 2 dims → [numSensors*numStim × numElems]
J = reshape(dfd, num_sensors*num_stim, n_elem);

return
end

function [Jx,Jy,Jz] = calc_jacobian_3axis(img)

mu0 = img.fwd_model.mu0;
%n_nodes = size(img.fwd_model.nodes,1);
n_elem = size(img.fwd_model.elems,1);

num_stim = numel(img.fwd_model.stimulation);
num_sensors = numel(img.fwd_model.sensors);

% Compute Gamma matrices
img = compute_gamma_matrices(img);

R = img.fwd_model.R;
G = img.fwd_model.G;

Gamma1 = img.Gamma1;
Gamma2 = img.Gamma2;
Gamma3 = img.Gamma3;

% Compute EIT forward solution for each current injection pattern, and
% grab the system matrix from the same call so system_mat_1st_order is
% assembled only once.
img.fwd_solve.get_all_meas = 1;
[data, A_matrix] = fwd_solve_sys_mat_mdeit(img);
u = data.volt;

Gamma1T = Gamma1.';
Gamma2T = Gamma2.';
Gamma3T = Gamma3.';

n_elec = numel(img.fwd_model.electrode);
Ac = A_matrix(1:size(img.fwd_model.nodes,1),1:size(img.fwd_model.nodes,1));
Ae = A_matrix(1:size(img.fwd_model.nodes,1),size(img.fwd_model.nodes,1)+1:end);
Aet = A_matrix(size(img.fwd_model.nodes,1)+1:end,1:size(img.fwd_model.nodes,1));
Ad = A_matrix(size(img.fwd_model.nodes,1)+1:end,size(img.fwd_model.nodes,1)+1:end);
Ad1 = sparse(1:n_elec,1:n_elec,1./diag(Ad),n_elec,n_elec);

A_matrix = Ac-Ae*Ad1*Aet;

% Solve the adjoint problem for each sensor to get lambda vectors
lambdaX = A_matrix \ (-Gamma1T);
lambdaY = A_matrix \ (-Gamma2T);
lambdaZ = A_matrix \ (-Gamma3T);


Gx_times_lambda_X = G.Gx*lambdaX;
Gy_times_lambda_X = G.Gy*lambdaX;
Gz_times_lambda_X = G.Gz*lambdaX;

Gx_times_lambda_Y = G.Gx*lambdaY;
Gy_times_lambda_Y = G.Gy*lambdaY;
Gz_times_lambda_Y = G.Gz*lambdaY;

Gx_times_lambda_Z = G.Gx*lambdaZ;
Gy_times_lambda_Z = G.Gy*lambdaZ;
Gz_times_lambda_Z = G.Gz*lambdaZ;

Gx_times_u = G.Gx*u;
Gy_times_u = G.Gy*u;
Gz_times_u = G.Gz*u;

mu_factor = mu0/(4*pi);

elemV = img.fwd_model.elem_volume(:);      % [numElems × 1]

% We want to broadcast arrays into num_sensors*num_stim*num_elems, in that order, so we avoid a permute in dfd

% Expand elem_volume to cover stim × sensor
elemV = reshape(elemV, [1 1 n_elem]);

% Expand u-terms to 3D
GxU = reshape(Gx_times_u.', [1 num_stim n_elem]); % [: × numStim × 1]
GyU = reshape(Gy_times_u.', [1 num_stim n_elem]);
GzU = reshape(Gz_times_u.', [1 num_stim n_elem]);

% Compute all dfdp (also 3D)
Rx_ = reshape(R.Rx, [num_sensors 1 n_elem]);
Ry_ = reshape(R.Ry, [num_sensors 1 n_elem]);
Rz_ = reshape(R.Rz, [num_sensors 1 n_elem]);

% These are the derivatives with respect to sigma of the C components of the Gamma matrix,
dCxdp = ( -Rz_.*GyU + Ry_.*GzU );
dCydp = ( -Rx_.*GzU + Rz_.*GxU );
dCzdp = ( -Ry_.*GxU + Rx_.*GyU );

% The g matrix does not depend on sigma.
g = zeros(num_sensors,3,3);
for m = 1:numel(img.fwd_model.sensors)
    g(m,:,:) = [...
        img.fwd_model.sensors(m).axes.axis1;
        img.fwd_model.sensors(m).axes.axis2;
        img.fwd_model.sensors(m).axes.axis3];
end

for select_sensor_axis = 1:3

    switch select_sensor_axis
        case 1
            % Expand lambda and R terms to 3D
            GxL = reshape(Gx_times_lambda_X.', [num_sensors 1 n_elem]);
            GyL = reshape(Gy_times_lambda_X.', [num_sensors 1 n_elem]);
            GzL = reshape(Gz_times_lambda_X.', [num_sensors 1 n_elem]);
        case 2
            % Expand lambda and R terms to 3D
            GxL = reshape(Gx_times_lambda_Y.', [num_sensors 1 n_elem]);
            GyL = reshape(Gy_times_lambda_Y.', [num_sensors 1 n_elem]);
            GzL = reshape(Gz_times_lambda_Y.', [num_sensors 1 n_elem]);
        case 3
            % Expand lambda and R terms to 3D
            GxL = reshape(Gx_times_lambda_Z.', [num_sensors 1 n_elem]);
            GyL = reshape(Gy_times_lambda_Z.', [num_sensors 1 n_elem]);
            GzL = reshape(Gz_times_lambda_Z.', [num_sensors 1 n_elem]);
    end

    % Compute all dfdx for all sensors+stim
    dfdx = elemV .* ( ...
        GxL.*GxU + ...
        GyL.*GyU + ...
        GzL.*GzU );



    % g: [num_sensors × 3 × 3]
    gx = reshape(g(:,select_sensor_axis,1), [num_sensors 1 1 ]);
    gy = reshape(g(:,select_sensor_axis,2), [num_sensors 1 1]);
    gz = reshape(g(:,select_sensor_axis,3), [num_sensors 1 1]);

    dfdp = mu_factor*(...
        gx.*dCxdp +...
        gy.*dCydp +...
        gz.*dCzdp);

    dfd = dfdx + dfdp;   % size: [numStim × numSensors × numElems]

    % Now reshape to match J(ids,:)

    % collapse first 2 dims → [numSensors*numStim × numElems]
    J = reshape(dfd, num_sensors*num_stim, n_elem);

    switch select_sensor_axis
        case 1
            Jx = J;
        case 2
            Jy = J;
        case 3
            Jz = J;
    end

end

return
end


%% FUNCTIONS
function F = factorise_symmetric(A)
F.kind = 'ldl';
try
    [F.L,F.D,F.P] = ldl(A,'vector');
    F.n = size(A,1);
catch
    error('Couldnt do it')
    % [F.L,F.U,F.pv,F.qv] = lu(A,'vector');
    % F.kind='lu';
    % F.n   = size(A,1);
end
end

function X = solve_fact_multiple_rhs(F, rhs)

switch F.kind

    case 'ldl'
        % Permute RHS (each column independently)
        rp = rhs(F.P, :);

        % LDL solves (all column-wise)
        y  = F.L \ rp;
        z  = F.D \ y;
        w  = F.L' \ z;

        % Allocate full solution matrix
        X = zeros(F.n, size(rhs,2));

        % Unpermute rows
        X(F.P, :) = w;

    case 'lu'
        % Row permutation of RHS
        y = rhs(F.pv, :);

        % Triangular solves
        z = F.L \ y;
        w = F.U \ z;

        % Allocate solution
        X = zeros(F.n, size(rhs,2));

        % Column permutation recovery
        X(F.qv, :) = w;

    otherwise
        error('Unknown factorisation kind.');
end
end


function recon_mode = check_recon_mode(img)

% Check which reconstruction mode is being used based on the sensor axes fields
if isfield(img.fwd_model.sensors(1).axes, 'axis')
    recon_mode = 'mdeit1';
elseif isfield(img.fwd_model.sensors(1).axes, 'axis1') && ...
        isfield(img.fwd_model.sensors(1).axes, 'axis2') && ...
        isfield(img.fwd_model.sensors(1).axes, 'axis3')
    recon_mode = 'mdeit3';
else
    error('Unknown sensor axes configuration. Please check the sensor axes fields.');
end

end