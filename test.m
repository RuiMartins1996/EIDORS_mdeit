clc; clear all; close all;

%% Prepare workspace

% Get the full path of the current script
fullpath = mfilename('fullpath');
% Extract just the folder
script_folder = fileparts(fullpath);
cd(script_folder);

% Have to add the functions path manually so prepare_workspace runs
addpath(genpath(fullfile(script_folder,'functions')));

eidors_folder = setup_eidors(script_folder);

%% Create forward model for reconstruction and for forward simulation

background_conductivity = 1.0;

height = 3;
radius = 1.0;
maxsz_recon = 0.4;
maxsz_fwd = 0.3;

num_electrodes_ring = 8;
num_rings = 4;
ring_vert_pos = height/(num_rings+1):height/(num_rings+1):height-height/(num_rings+1);
electrode_radius = 0.3;

z_contact_impedance = 1.0;
current_amplitude = 1.0;

elec_pos = [num_electrodes_ring,ring_vert_pos];
elec_shape = [electrode_radius, electrode_radius, maxsz_recon ]; %square electrodes

anomaly_conductivity = 0.1; % (0 according to Kai's thesis, but check notes)
anomaly_position = [radius/2,0,height/2]; %  [0,0,35]*1e-3/l0; % pg 172
anomaly_radius = 0.25; % 25mm of diameter, pg 172

% Create forward models
fmdl_recon = ng_mk_cyl_models([height,radius,maxsz_recon],elec_pos,elec_shape);
fmdl_fwd = ng_mk_cyl_models([height,radius,maxsz_fwd],elec_pos,elec_shape);

show_fem(fmdl_recon)

%% Create stimulation pattern

stim = mk_stim_patterns(num_electrodes_ring,num_rings,[0,1],[0,1],{'meas_current'},current_amplitude);

fmdl_recon.stimulation = stim;
fmdl_fwd.stimulation = stim;

for i = 1:num_electrodes_ring*num_rings
    fmdl_recon.electrode(i).z_contact = z_contact_impedance;
    fmdl_fwd.electrode(i).z_contact = z_contact_impedance;
end
%% Assign magnetometers

num_sensors_ring = num_electrodes_ring;
sensor_positions = zeros(numel(fmdl_recon.electrode),3);
sensor_distance = radius/5;

cyl_center = [0,0,height/2];

for m = 1:numel(fmdl_recon.electrode)
    elec_nodes = fmdl_recon.nodes(fmdl_recon.electrode(m).nodes,:);
    elec_center = mean(elec_nodes,1);
    
    displacement = elec_center-cyl_center;
    displacement = [displacement(1:2),0];
    displacement = displacement/norm(displacement);

    sensor_positions(m,:) = elec_center + displacement*sensor_distance;
end

sensor_axes = struct('axis1',[1,0,0],'axis2',[0,1,0],'axis3',[0,0,1]);

fmdl_recon = assign_magnetometers(fmdl_recon,sensor_positions,sensor_axes);
fmdl_fwd = assign_magnetometers(fmdl_fwd,sensor_positions,sensor_axes);

plot_sensors(fmdl_recon);
%% Initialize model (NEED IMPROVEMENT?)

mu0 = 1;
fmdl_recon = compute_geometry_matrices(fmdl_recon,mu0);
fmdl_fwd = compute_geometry_matrices(fmdl_fwd,mu0);


%% Make homogeneous/inhomogeneous images

img1 = mk_image(fmdl_fwd,background_conductivity);

select_fcn = @(x,y,z) (x-anomaly_position(1)).^2 + (y-anomaly_position(2)).^2 + (z-anomaly_position(3)).^2 < anomaly_radius^2;
memb_frac = elem_select( img1.fwd_model, select_fcn);
img2 = mk_image(img1, background_conductivity*(1 + memb_frac));

%% Test jacobian vector product

% Uncomment to verify the operator against your existing dense
% calc_jacobian_mdeit on a case small enough to assemble J explicitly:

img = img1;
J   = calc_jacobian_mdeit(img);
Jop = calc_jacobian_operator_mdeit(img);

for i = 1:10
v   = randn(size(J,2),1);
w   = randn(size(J,1),1);
disp(norm(J*v      - Jop.mtimes(v))       / norm(J*v));
disp(norm(J.'*w    - Jop.mtimes_transp(w))/ norm(J.'*w));
end

% Both should be ~1e-12 (machine precision), confirming the
% contraction-order rewrite is mathematically identical to the
% explicit tensor construction.


%% Compute forward solutions 

% Compute forward solutions
datah_mdeit = fwd_solve_1st_order_mdeit(img1);
datai_mdeit = fwd_solve_1st_order_mdeit(img2);

datah_eit = fwd_solve(img1);
datai_eit = fwd_solve(img2);

%% Pollute data with a small amount of gaussian white noise

% Add gaussian white noise
data = calc_difference_data_mdeit(datah_mdeit,datai_mdeit,img1.fwd_model);
noise_strength = max(abs(data))/10; % Default noise strength

% datah_mdeit.meas = datah_mdeit.meas + noise_level_mdeit * randn(size(datah_mdeit.meas));
datai_mdeit.meas = datai_mdeit.meas + noise_strength * randn(size(datai_mdeit.meas));

noise_level_eit = max([datah_eit.meas;datai_eit.meas])*noise_strength;
datah_eit.meas = datah_eit.meas + noise_level_eit * randn(size(datah_eit.meas));
datai_eit.meas = datai_eit.meas + noise_level_eit * randn(size(datai_eit.meas));

%% Create inverse model

imdl = mk_common_model('a2c2',8); % Will replace most fields
imdl.fwd_model = fmdl_recon;
imdl = rmfield(imdl,'jacobian_bkgnd');

img_bkgnd = mk_image(imdl,background_conductivity);
imdl.jacobian_bkgnd.elem_data = img_bkgnd.elem_data;

% imdl.RtR_prior = @prior_tikhonov;% the default prior is prior_laplace
imdl.RtR_prior = @prior_tikhonov;% the default prior is prior_laplace

%% MDEIT solve 

imdl.inv_solve_core.print_diagnostics = true;

imdl.hyperparameter.value = 0.005;
imgr_mdeit = inv_solve_diff_GN_one_step_mdeit(imdl, datah_mdeit,datai_mdeit);

%% Noise correction

function noise_data = noise_generator(imgh,imgi,num_noise_repetitions)
datah = fwd_solve_1st_order_mdeit(imgh);
datai = fwd_solve_1st_order_mdeit(imgi);

data = calc_difference_data_mdeit(datah,datai,imgh.fwd_model);

% Add Gaussian white noise
noise_strength = max(abs(data))/10; % Default noise strength
noise_data = noise_strength * randn(size(data,1),num_noise_repetitions);

end

num_noise_repetitions = 30;
noise_data= noise_generator(img1,img2,num_noise_repetitions);

sigma_std = noise_correction(imdl,noise_data);

imgr_mdeit_noise_corrected = imgr_mdeit;
imgr_mdeit_noise_corrected.elem_data = imgr_mdeit.elem_data./sigma_std;

%% Plots 
figure

subplot(1,2,1)
show_fem(imgr_mdeit)
plot_sensors(imgr_mdeit)
title('Normal')

subplot(1,2,2)
show_fem(imgr_mdeit_noise_corrected)
plot_sensors(imgr_mdeit_noise_corrected)
title('Noise-Corrected')

disp('Done');

