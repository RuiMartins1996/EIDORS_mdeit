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

%% Create forward model

height = 3;
radius = 1;
maxsz = 0.2;

num_electrodes_ring = 8;
num_rings = 4;
ring_vert_pos = height/(num_rings+1):height/(num_rings+1):height-height/(num_rings+1);
electrode_radius = 0.2;

cyl_shape = [height,radius,maxsz];
elec_pos = [num_electrodes_ring,ring_vert_pos];
elec_shape = [electrode_radius, electrode_radius, maxsz ]; %square electrodes

fmdl= ng_mk_cyl_models(cyl_shape,elec_pos,elec_shape);

show_fem(fmdl)

%% Create stimulation pattern

stim = mk_stim_patterns(num_electrodes_ring,num_rings,[0,1],[0,1],{'meas_current'},1);
fmdl.stimulation = stim;

%% Assign magnetometers

num_sensors_ring = num_electrodes_ring;
sensor_positions = zeros(numel(fmdl.electrode),3);
sensor_distance = 0.1;

cyl_center = [0,0,height/2];

for m = 1:numel(fmdl.electrode)
    elec_nodes = fmdl.nodes(fmdl.electrode(m).nodes,:);
    elec_center = mean(elec_nodes,1);
    
    displacement = elec_center-cyl_center;
    displacement = [displacement(1:2),0];
    displacement = displacement/norm(displacement);

    sensor_positions(m,:) = elec_center + displacement*sensor_distance;
end

sensor_axes = struct('axis1',[1,0,0],'axis2',[0,1,0],'axis3',[0,0,1]);
fmdl = assign_magnetometers(fmdl,sensor_positions,sensor_axes);

plot_sensors(fmdl);
%% Initialize model (NEED IMPROVEMENT?)

mu0 = 1;
fmdl = compute_geometry_matrices(fmdl,mu0);

%% Solve model

imdl = mk_common_model('a2c2',8); % Will replace most fields
imdl.fwd_model = fmdl;
imdl.fwd_model.stimulation = fmdl.stimulation;
imdl.hyperparameter.value = 0.001;
imdl.RtR_prior = @prior_tikhonov;

img1 = mk_image(imdl);

% Add a circular object at 0.2, 0.5
% Calculate element membership in object
select_fcn = @(x,y,z) (x-0).^2 + (y-0).^2 + (z-1.5).^2 < 0.3^2;
memb_frac = elem_select( img1.fwd_model, select_fcn);
img2 = mk_image(img1, 1 + memb_frac );

% the default prior is prior_laplace

% Simulate Voltages and plot them
datah_mdeit = fwd_solve_1st_order_mdeit(img1);
datai_mdeit = fwd_solve_1st_order_mdeit(img2);

datah_eit = fwd_solve(img1);
datai_eit = fwd_solve(img2);

imgr_eit = inv_solve(imdl,datah_eit,datai_eit);

imdl.hyperparameter.value = 0.01;
imgr_mdeit = inv_solve_diff_GN_one_step_mdeit(imdl, datah_mdeit, datai_mdeit);


figure
subplot(1,2,1)
show_fem(imgr_eit)
subplot(1,2,2)
show_fem(imgr_mdeit)

