function sigma_std = noise_correction(inv_model,noise_data,hp)

% Check if hyperparameter hp is provided
if nargin < 3
    % Check if inv_model has a hyperparameter field
    if isfield(inv_model, 'hyperparameter') && isfield(inv_model.hyperparameter, 'value')
        hp = inv_model.hyperparameter.value;
    else
        error('noise_correction: inv_model.hyperparameter.value must be provided if not input argument');
    end
end

recon_mode = check_mdeit_reconstruction_mode(inv_model.fwd_model);
switch recon_mode
    case 'mdeit1'
        num_meas_per_sensor = 1;
    case 'mdeit3'
        num_meas_per_sensor = 3;
    otherwise
        error('noise_correction: unrecognized reconstruction mode');
end

% Check that num_meas is consistent with fwd_model
num_meas = size(noise_data,1);
for i = 1:numel(inv_model.fwd_model.stimulation)
    if num_meas ~= size(inv_model.fwd_model.stimulation(i).meas_pattern,1)*numel(inv_model.fwd_model.sensors)*num_meas_per_sensor
        error('noise_correction: Number of measurements is inconsistent.');
    end
end

% Compute the Jacobian for the homogeneous image
img_bkgnd = calc_jacobian_bkgnd(inv_model);
Jh = calc_jacobian_mdeit(img_bkgnd);

% Compute the SVD of the Jacobian
fprintf('Computing SVD of the Jacobian...\n');
[U,S,V] = svd(Jh,'econ');

s  = diag(S);
sv = s + hp./s;
M = V * diag(1./sv) * U' * noise_data;

sigma_std = std(M,[],2);


end


function recon_mode =  check_mdeit_reconstruction_mode(fwd_model)

% Check if field sensors.axes exists
if ~isfield(fwd_model.sensors(1), 'axes')
    error('noise_correction: fwd_model.sensors must have axes field');
end

% Check if all the sensors have the same reconstruction mode
for i = 1:numel(fwd_model.sensors)
    if ~isequal(fwd_model.sensors(1).axes, fwd_model.sensors(i).axes)
        error('noise_correction: All sensors must have the same reconstruction mode');
    end
end

% Check if the reconstruction mode is one-axis MDEIT or three-axis MDEIT
if isfield(fwd_model.sensors(1).axes, 'axis')
    recon_mode = 'mdeit1';
elseif isfield(fwd_model.sensors(1).axes, 'axis1') && ...
        isfield(fwd_model.sensors(1).axes, 'axis2') && ...
        isfield(fwd_model.sensors(1).axes, 'axis3')
    recon_mode = 'mdeit3';
else
    error('noise_correction: unrecognized field in fwd_model.sensors');
end

end