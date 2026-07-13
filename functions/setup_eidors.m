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
% helper: searches upward for a folder containing "eidors" in its name.
% At each ancestor level, both that level's children AND grandchildren are
% checked (e.g. .../Scripts/eidors-v3.12-ng is a grandchild when starting
% from .../EIDORS_mdeit/tests, since eidors-v3.12-ng sits under the sibling
% folder "Scripts", not directly under a common ancestor).

    max_levels = 5;
    current = start_folder;

    for k = 1:max_levels

        candidates = list_subdirs(current);
        grandchildren = {};
        for i = 1:numel(candidates)
            grandchildren = [grandchildren, list_subdirs(candidates{i})]; %#ok<AGROW>
        end
        candidates = [candidates, grandchildren];

        % Look for "*eidors*" (case-insensitive) among candidates. A name
        % match alone is not enough: "EIDORS_mdeit" (this project's own
        % folder) also contains "eidors" and can shadow the real EIDORS
        % install if it happens to sort earlier than "Scripts". Require the
        % candidate to actually contain eidors/startup.m.
        [~,names] = cellfun(@fileparts, candidates, 'UniformOutput', false);
        idx = find(contains(lower(names), 'eidors'));

        for k2 = 1:numel(idx)
            candidate = candidates{idx(k2)};
            if exist(fullfile(candidate, 'eidors', 'startup.m'), 'file')
                eidors_folder = candidate;
                return;
            end
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

function subdirs = list_subdirs(folder)
    d = dir(folder);
    isub = [d.isdir];
    names = {d(isub).name};
    names = names(~ismember(names,{'.','..'}));
    subdirs = cellfun(@(n) fullfile(folder,n), names, 'UniformOutput', false);
end


