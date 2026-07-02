function img = compute_gamma_matrices(img)

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

switch recon_mode
    case 'mdeit1'
        img = compute_gamma_matrices_mdeit1(img);
    case 'mdeit3'
        img = compute_gamma_matrices_mdeit3(img);
    otherwise
        error('Unknown reconstruction mode: %s', recon_mode);
end

end

function img = compute_gamma_matrices_mdeit1(img)
    mu_factor = img.fwd_model.mu0/(4*pi);

    num_sensors = numel(img.fwd_model.sensors);

    % Convenience handles
    R = img.fwd_model.R;
    G = img.fwd_model.G;

    Sigma = spdiags(img.elem_data(:), 0, length(img.elem_data), length(img.elem_data));

    g = zeros(num_sensors, 3);
    for m = 1:numel(img.fwd_model.sensors)
        g(m,:) = img.fwd_model.sensors(m).axes.axis;
    end

    Cx = ( -R.Rz * Sigma * G.Gy +  R.Ry * Sigma * G.Gz );
    Cy = ( -R.Rx * Sigma * G.Gz +  R.Rz * Sigma * G.Gx );
    Cz = ( -R.Ry * Sigma * G.Gx +  R.Rx * Sigma * G.Gy );

    Gamma = mu_factor * (g(:,1).*Cx + g(:,2).*Cy + g(:,3).*Cz);

    img.Gamma = Gamma;
end

function img = compute_gamma_matrices_mdeit3(img)
mu_factor = img.fwd_model.mu0/(4*pi);

num_sensors = numel(img.fwd_model.sensors);

% Convenience handles
R = img.fwd_model.R;
G = img.fwd_model.G;

% Sigma = sparse(1:length(img.elem_data), 1:length(img.elem_data), img.elem_data);
Sigma = spdiags(img.elem_data(:), 0, length(img.elem_data), length(img.elem_data));

% NEW: The matrix g_{dl}^m, gives the components of the measurement axis of
% sensor m on the canonical R^3 basis

g = zeros(num_sensors,3,3);
for m = 1:numel(img.fwd_model.sensors)
    g(m,:,:) = [...
        img.fwd_model.sensors(m).axes.axis1;
        img.fwd_model.sensors(m).axes.axis2;
        img.fwd_model.sensors(m).axes.axis3];
end

% Check if Gamma fields already exist and are the correct size
check_gamma(img);

Cx = ( -R.Rz * Sigma * G.Gy +  R.Ry * Sigma * G.Gz );
Cy = ( -R.Rx * Sigma * G.Gz +  R.Rz * Sigma * G.Gx );
Cz = ( -R.Ry * Sigma * G.Gx +  R.Rx * Sigma * G.Gy );

Gamma1 = mu_factor*(g(:,1,1).*Cx + g(:,1,2).*Cy + g(:,1,3).*Cz);
Gamma2 = mu_factor*(g(:,2,1).*Cx + g(:,2,2).*Cy + g(:,2,3).*Cz);
Gamma3 = mu_factor*(g(:,3,1).*Cx + g(:,3,2).*Cy + g(:,3,3).*Cz);

img.Gamma1 = Gamma1;
img.Gamma2 = Gamma2;
img.Gamma3 = Gamma3;

end


% Check if Gamma fields already exist and are the correct size
function check_gamma(img)

num_sensors = numel(img.fwd_model.sensors);
num_nodes   = size(img.fwd_model.nodes, 1);
expectedSize = [num_sensors, num_nodes];

% Check for existing fields and reuse if available
if isfield(img, 'Gamma1') && ~isempty(img.Gamma1)
    if not(isequal(size(img.Gamma1), expectedSize))
        error('Expected size for Gamma matrix is wrong')
    end
end

% Check for existing fields and reuse if available
if isfield(img, 'Gamma2') && ~isempty(img.Gamma2)
    if not(isequal(size(img.Gamma2), expectedSize))
        error('Expected size for Gamma matrix is wrong')
    end
end


% Check for existing fields and reuse if available
if isfield(img, 'Gamma3') && ~isempty(img.Gamma3)
    if not(isequal(size(img.Gamma3), expectedSize))
        error('Expected size for Gamma matrix is wrong')
    end
end

end