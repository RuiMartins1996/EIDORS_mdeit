function eidors_folder = setup_eidors(start_folder)

% Validate input
if nargin < 1 || ~isfolder(start_folder)
    error('start_folder must be a valid directory.');
end

% Require the Parallel Computing Toolbox before continuing
if ~has_parallel_computing_toolbox()
    error(['Parallel Computing Toolbox is required to run this project. ' ...
        'Please install and license it before continuing.']);
end

% Search parent directories for an EIDORS directory 
eidors_folder = find_eidors_folder(start_folder);

if isempty(eidors_folder)
    error('Could not locate an EIDORS folder in parent directories.');
end

fprintf('Found EIDORS at: %s\n', eidors_folder);

% Add EIDORS to path
addpath(genpath(eidors_folder));

% Run startup.m
startup_file = fullfile(eidors_folder, 'eidors', 'startup.m');

if exist(startup_file, 'file')
    run(startup_file);
else
    error('startup.m was not found at %s', startup_file);
end

end

function tf = has_parallel_computing_toolbox()
% Detect whether MATLAB has access to the Parallel Computing Toolbox.

tf = false;

if exist('license', 'builtin') || exist('license', 'file')
    try
        tf = license('test', 'Distrib_Computing_Toolbox');
    catch
        tf = false;
    end
end

if ~tf && (exist('ver', 'builtin') || exist('ver', 'file'))
    try
        installed_products = ver;
        tf = any(strcmpi({installed_products.Name}, 'Parallel Computing Toolbox'));
    catch
        tf = false;
    end
end

end

function eidors_folder = find_eidors_folder(start_folder)
% helper: searches upward for a folder containing "eidors" in its name

    max_levels = 5;
    current = start_folder;

    for k = 1:max_levels

        % List subfolders
        d = dir(current);
        isub = [d.isdir];
        subdirs = {d(isub).name};
        subdirs = subdirs(~ismember(subdirs,{'.','..'}));

        % Look for "*eidors*" (case-insensitive)
        idx = find(contains(lower(subdirs), 'eidors'));

        if ~isempty(idx)
            eidors_folder = fullfile(current, subdirs{idx(1)});
            return;
        end

        % Move one level up
        parent = fileparts(current);
        if strcmp(parent, current)
            break; % reached filesystem root
        end
        current = parent;
    end

    % Nothing found
    eidors_folder = '';
end


