clc; clear all; close all;

%% Prepare workspace

% Get the full path of the current script
fullpath = mfilename('fullpath');
% Extract just the folder
script_folder = fileparts(fullpath);
cd(script_folder);

% Have to add the functions path manually so prepare_workspace runs
addpath(genpath(fullfile(fileparts(script_folder),'functions')));

eidors_folder = setup_eidors(script_folder);

%% Create circular forward model

radius = 1;
n_electrodes = 16;
electrode_radius = 0.15;
maxsz = 0.05;

fmdl= ng_mk_cyl_models([0,radius,maxsz],n_electrodes,[electrode_radius, 0, maxsz ]);

show_fem(fmdl)

%% Make stimulation pattern
stim = mk_stim_patterns(n_electrodes,1,[0,1],[0,1],{'meas_current'},1);
fmdl.stimulation = stim;

%% Assign magnetometers

select_sensor_axis = 3;
num_sensors_ring = n_electrodes;
sensor_positions = zeros(num_sensors_ring,3);
sensor_distance = radius + 0.1; % Place sensors slightly outside the cylinder

for i = 1:num_sensors_ring
    angle = 2*pi*(i-1)/num_sensors_ring;
    sensor_positions(i,:) = [sensor_distance*cos(angle), sensor_distance*sin(angle), 0];
end

sensor_axes = struct('axis',[0,0,1]);
fmdl = assign_magnetometers(fmdl,sensor_positions,sensor_axes);

plot_sensors(fmdl);


%% Initialize model (NEED IMPROVEMENT?)

mu0 = 1;
fmdl = compute_geometry_matrices(fmdl,mu0);

%% Make homogeneous/inhomogeneous images
imgh = mk_image(fmdl,1.0);

% Add a circular object at 0.2, 0.5
% Calculate element membership in object
select_fcn = @(x,y,z) (x-0).^2 + (y-0).^2 < 0.3^2;
memb_frac = elem_select( imgh.fwd_model, select_fcn);
imgi = mk_image(imgh, 1 + memb_frac );

%% Compute forward solutions and add noise

datah_mdeit = fwd_solve_1st_order_mdeit(imgh);
datai_mdeit = fwd_solve_1st_order_mdeit(imgi);

datah_eit = fwd_solve(imgh);
datai_eit = fwd_solve(imgi);

% Pollute data with a small amount of gaussian white noise
noise_strength = 0.00001;
noise_level_mdeit = max([datah_mdeit.meas;datai_mdeit.meas])*noise_strength;
datah_mdeit.meas = datah_mdeit.meas + noise_level_mdeit * randn(size(datah_mdeit.meas));
datai_mdeit.meas = datai_mdeit.meas + noise_level_mdeit * randn(size(datai_mdeit.meas));

noise_level_eit = max([datah_eit.meas;datai_eit.meas])*noise_strength;
datah_eit.meas = datah_eit.meas + noise_level_eit * randn(size(datah_eit.meas));
datai_eit.meas = datai_eit.meas + noise_level_eit * randn(size(datai_eit.meas));

%% Create inverse model

imdl = mk_common_model('a2c2',8); % Will replace most fields
imdl.fwd_model = fmdl;

% imdl.RtR_prior = @prior_tikhonov;% the default prior is prior_laplace
imdl.RtR_prior = @prior_noser;% the default prior is prior_laplace

%% EIT solve
imdl.hyperparameter.value = 0.001;
imgr_eit = inv_solve(imdl,datah_eit,datai_eit);

%% MDEIT solve
% Use GCV to find the optimal hyperparameter
% imdl.hyperparameter.func = @gcv;
if isfield(imdl.hyperparameter,'func')
    imdl.hyperparameter = rmfield(imdl.hyperparameter,'func');
end
imdl.hyperparameter.value = 0.0005;
% Use pcg solve
% imdl.inv_solve_core.do_pcg = true;

imgr_mdeit = inv_solve_diff_GN_one_step_mdeit(imdl, datah_mdeit, datai_mdeit);

%% Plots 
figure
plot(1,1)

figure
subplot(1,2,1)
show_fem(imgr_eit)
subplot(1,2,2)
show_fem(imgr_mdeit)
plot_sensors(imgr_mdeit)

disp('Done');