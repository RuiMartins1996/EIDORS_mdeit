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

%% Define the characteristic scales in SI units

z0 = 0.0058; %(Ohm m^2) is the contact impedance from the CEM article 58 Ohm cm^2
l0 = 40e-3; %(m) the tank radius
I0 = 2.4e-3;%(A) the magnitude of the injected current

% The derived characteristic units
V0 = z0*I0/(l0^2); %(V)
sigma0 = l0/z0; %(S/m)
J0 = I0/(l0^2);

%% Create forward model for reconstruction and for forward simulation

background_conductivity = 3.28e-1/sigma0;

height = 70e-3/l0;
radius = 40e-3/l0;
maxsz_recon = 5e-2/l0;
maxsz_fwd = 10e-3/l0;

num_electrodes_ring = 8;
num_rings = 4;
ring_vert_pos = height/(num_rings+1):height/(num_rings+1):height-height/(num_rings+1);
electrode_radius = 8e-3/l0;

z_contact_impedance = 0.0058/z0;
current_amplitude = 2.4e-3/I0;

elec_pos = [num_electrodes_ring,ring_vert_pos];
elec_shape = [electrode_radius, electrode_radius, maxsz_recon ]; %square electrodes

anomaly_conductivity = 1e-12/sigma0; % (0 according to Kai's thesis, but check notes)
anomaly_position = [20,0,35]*1e-3/l0; %  [0,0,35]*1e-3/l0; % pg 172
anomaly_radius = 25/2*1e-3/l0; % 25mm of diameter, pg 172

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
sensor_distance = 70e-3/l0-radius;

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

%% Compute forward solutions and add noise

% Compute forward solutions
datah_mdeit = fwd_solve_1st_order_mdeit(img1);
datai_mdeit = fwd_solve_1st_order_mdeit(img2);

datah_eit = fwd_solve(img1);
datai_eit = fwd_solve(img2);

% Pollute data with a small amount of gaussian white noise
noise_strength = 0.001;
noise_level_mdeit = max([datah_mdeit.meas;datai_mdeit.meas])*noise_strength;
datah_mdeit.meas = datah_mdeit.meas + noise_level_mdeit * randn(size(datah_mdeit.meas));
datai_mdeit.meas = datai_mdeit.meas + noise_level_mdeit * randn(size(datai_mdeit.meas));

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

%% MDEIT solve (with multiple ways to choose ridge parameter)

% Use GCV to find the optimal hyperparameter
imdl.inv_solve_core.print_diagnostics = true;
imgr_mdeit_gcv = inv_solve_absolute_GN_mdeit(imdl, datai_mdeit);


%% Plots 
figure

subplot(1,3,1)
title('Value')
show_fem(imgr_mdeit_value)
plot_sensors(imgr_mdeit_value)
subplot(1,3,2)
title('L-curve')
show_fem(imgr_mdeit_l_curve)
plot_sensors(imgr_mdeit_l_curve)
subplot(1,3,3)
title('GCV')
show_fem(imgr_mdeit_gcv)
plot_sensors(imgr_mdeit_gcv)

disp('Done');