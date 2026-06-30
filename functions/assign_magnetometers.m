function fmdl = assign_magnetometers(fmdl,sensor_positions,meas_axes)
%ASSIGN_MAGNETOMETERS Attach magnetometer sensors to a forward model.
%   fmdl = ASSIGN_MAGNETOMETERS(fmdl, sensor_positions, meas_axes)
%
%   sensor_positions: num_sensors x 3, one row [x,y,z] per sensor.
%
%   meas_axes (optional) sets each sensor's axis1/axis2/axis3, accepted as:
%     - omitted or []   : default Cartesian axes for every sensor
%     - 3x3 numeric     : one set of axes (rows = axis1,axis2,axis3),
%                         broadcast to all sensors 
%     - struct (1x1)    : fields axis1/axis2/axis3 (1x3 each),
%                         broadcast to all sensors
%     - struct (1xN)    : N must equal num_sensors; one set per sensor
%
%   All axes are normalized to unit length (zero vectors error out).
%
%   Output: fmdl.sensors(m).position / .axes (struct with normalized
%   axis1, axis2, axis3).

if size(sensor_positions,2) ~= 3
    error('sensor_positions must be num_sensors x 3 matrix');
end
num_sensors = size(sensor_positions,1);

if nargin < 3 || isempty(meas_axes)
    % Default: Cartesian axes for every sensor
    meas_axes = repmat(struct('axis1',[1,0,0],'axis2',[0,1,0],'axis3',[0,0,1]), 1, num_sensors);
else
    meas_axes = parse_meas_axes(meas_axes, num_sensors);
end

% Assign sensor locations and sensor axis to fwd_model
sensors = repmat(struct('position', [], 'axes', []), 1, num_sensors);
for m = 1:num_sensors
    sensors(m).position = sensor_positions(m,:);
    sensors(m).axes = meas_axes(m);
end
fmdl.sensors = sensors;
end


function meas_axes = parse_meas_axes(meas_axes, num_sensors)
if isnumeric(meas_axes)
    if ~isequal(size(meas_axes), [3,3])
        error('Numeric meas_axes must be a 3x3 matrix (rows = axis1, axis2, axis3)');
    end
    % Convert the single 3x3 axis matrix into a struct, then broadcast
    axis_struct = struct('axis1', meas_axes(1,:), ...
                          'axis2', meas_axes(2,:), ...
                          'axis3', meas_axes(3,:));
    meas_axes = repmat(axis_struct, 1, num_sensors);
else
    validate_meas_axes_struct(meas_axes);
    if numel(meas_axes) == 1
        meas_axes = repmat(meas_axes, 1, num_sensors);
    elseif numel(meas_axes) ~= num_sensors
        error('meas_axes must have length 1 or equal to num_sensors');
    end
end

% Normalize every axis vector for every sensor
required_fields = {'axis1','axis2','axis3'};
for m = 1:numel(meas_axes)
    for f = 1:numel(required_fields)
        v = meas_axes(m).(required_fields{f});
        n = norm(v);
        if n == 0
            error('meas_axes(%d).%s is a zero vector and cannot be normalized', m, required_fields{f});
        end
        meas_axes(m).(required_fields{f}) = v / n;
    end
end
end

function validate_meas_axes_struct(meas_axes)
if ~isstruct(meas_axes)
    error('meas_axes must be a struct array (with fields axis1, axis2, axis3) or a 3x3 numeric matrix');
end
required_fields = {'axis1','axis2','axis3'};
missing = required_fields(~isfield(meas_axes, required_fields));
if ~isempty(missing)
    error('meas_axes is missing required field(s): %s', strjoin(missing, ', '));
end
for m = 1:numel(meas_axes)
    for f = 1:numel(required_fields)
        v = meas_axes(m).(required_fields{f});
        if numel(v) ~= 3 || ~isnumeric(v)
            error('meas_axes(%d).%s must be a 1x3 numeric vector', m, required_fields{f});
        end
    end
end
end