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
maxsz = 0.2;

num_electrodes_ring = 6;
num_rings = 2;
ring_vert_pos = height/(num_rings+1):height/(num_rings+1):height-height/(num_rings+1);
electrode_radius = 0.3;

z_contact_impedance = 1.0;
current_amplitude = 1.0;

elec_pos = [num_electrodes_ring,ring_vert_pos];
elec_shape = [electrode_radius, electrode_radius, maxsz ]; %square electrodes

anomaly_conductivity = 10; % (0 according to Kai's thesis, but check notes)
anomaly_position = [radius/2,0,height/2]; %  [0,0,35]*1e-3/l0; % pg 172
anomaly_radius = 0.25; % 25mm of diameter, pg 172

% Create forward model
% Absolute reconstruction is very unstable to modelling error, se we will commit an inverse crime here.
% In specific, we simulate the forward data with the same model that we will use for reconstruction.
fmdl = ng_mk_cyl_models([height,radius,maxsz],elec_pos,elec_shape);
%% Create stimulation pattern

stim = mk_stim_patterns(num_electrodes_ring,num_rings,[0,1],[0,1],{'meas_current'},current_amplitude);

fmdl.stimulation = stim;

for i = 1:num_electrodes_ring*num_rings
    fmdl.electrode(i).z_contact = z_contact_impedance;
end
%% Assign magnetometers

num_sensors_ring = num_electrodes_ring;
sensor_positions = zeros(numel(fmdl.electrode),3);

num_sensors = num_sensors_ring*num_rings;

thetam = 0:2*pi/num_sensors_ring:2*pi-2*pi/num_sensors_ring;
sensor_radius = 1.5*radius;

for i = 1:num_rings
    for m = 1:num_sensors_ring
        
        xm = sensor_radius*cos(thetam(m));
        ym = sensor_radius*sin(thetam(m));
        zm = ring_vert_pos(i);

        sensor_positions((i-1)*num_sensors_ring+m,:) = [xm,ym,zm];
    end
end

sensor_axes = struct('axis1',[1,0,0],'axis2',[0,1,0],'axis3',[0,0,1]);

fmdl = assign_magnetometers(fmdl,sensor_positions,sensor_axes);
%% Initialize model (NEED IMPROVEMENT?)

mu0 = 1;
fmdl = compute_geometry_matrices(fmdl,mu0);


%% Make homogeneous/inhomogeneous images

img1 = mk_image(fmdl,background_conductivity);

select_fcn = @(x,y,z) (x-anomaly_position(1)).^2 + (y-anomaly_position(2)).^2 + (z-anomaly_position(3)).^2 < anomaly_radius^2;
memb_frac = elem_select( img1.fwd_model, select_fcn);
img2 = mk_image(img1, background_conductivity*(1 + memb_frac));

%% Compute forward solutions 

% Compute forward solutions
%datah_mdeit = fwd_solve_1st_order_mdeit(img1);
datai_mdeit = fwd_solve_1st_order_mdeit(img2);

%datah_eit = fwd_solve(img1);
datai_eit = fwd_solve(img2);

% Do noiseless reconstruction
%% Create inverse model

imdl = mk_common_model('a2c2',8); 
imdl.fwd_model = fmdl;
imdl = rmfield(imdl,'jacobian_bkgnd');

% Setup inverse model
img_bkgnd = mk_image(imdl,background_conductivity); 
% In absolute reconstruction, the background conductivity is used as prior.
imdl.jacobian_bkgnd.elem_data = img_bkgnd.elem_data; 
imdl.RtR_prior = @prior_noser;

% Setup solver parameters
imdl.inv_solve_core.print_diagnostics = false;
imdl.inv_solve_core.do_pcg = true;
imdl.hyperparameter.value = 0.005;

%% EIT inverse solve 
[img_eit_absolute_gn] = inv_solve_abs_GN_prior( imdl, datai_eit);
% [img_eit_absolute_gn_c] = inv_solve_abs_GN_constrain(imdl, datai_eit);

%% MDEIT inverse solve
imgr_mdeit_absolute_gn = inv_solve_absolute_GN_mdeit(imdl, datai_mdeit);

%% Plots 
figure

subplot(1,2,1)
show_fem(img_eit_absolute_gn);
plot_sensors(img_eit_absolute_gn);
title('EIT: Absolute GN')

subplot(1,2,2)
show_fem(imgr_mdeit_absolute_gn);
plot_sensors(imgr_mdeit_absolute_gn);
title('MDEIT: Absolute GN')

disp('Done');