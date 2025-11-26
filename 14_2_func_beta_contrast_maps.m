%%% Script developped to calculate contrasts
%participant = 'sub-NSxGxHKx1965';


%% Variables
glmMode = 'surface';
TR = 2; % seconds

%%% define contrasts by task
% contrasts.ptlocal = 'motion = 1 : stationary = -1';
% contrasts.mtlocal = 'motion = 1 : stationary = -1';

contrasts.ptlocal = 'motion = 1 : silent = -1';
contrasts.mtlocal = 'motion = 1 : silent = -1';
% ensure mode is vaild
% switch glmMode
%     case {'surface', 'volume'}
%         glmDir = sprintf('GLMsingle-%s', glmMode);
%     otherwise
%         error('Unrecognized mode selected.')
% end
%% Directories 
paths.main = fullfile('/Users','ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb'); 
paths.data = fullfile(paths.main); 

participants = readlines(fullfile(paths.main, 'code', 'utils', 'study2_subjects_updated.txt'));

for n =1:length(participants)
    participant = participants{n};
    paths.raw  = fullfile(paths.data, participant);
    paths.func = fullfile(paths.data, 'analysis', 'fMRI_data', participant, 'func');
    paths.save = fullfile(paths.main, 'analysis', 'fMRI_data', participant, 'glm');
    if ~isfolder(paths.save); mkdir(paths.save); end

    clear list; % clear previous

    % locate all bold surface files
    switch glmMode
        case 'surface'
            boldPattern = '*_bold.func.gii';
            list.hemi = {'hemi-L', 'hemi-R'};
        case 'volume'
            boldPattern = '*_space-T1w_desc-preproc_bold.nii.gz';
    end
    boldFiles  = dir(fullfile(paths.func, boldPattern));
    boldFiles  = arrayfun(@(x) fullfile(x.folder, x.name), boldFiles, ...
        'UniformOutput', false);

    taskList = regexprep(boldFiles, '.+_task-(\w+)_.+', '$1');
    list.task = unique(taskList); % unique task list
    conds = combvec_conditions(list);

    %Compute contrasts

    for i = 3:length(conds) % for each task
        % subset bold files by task types
        indx = struct_contains(boldFiles, conds(i));
        currBold = boldFiles(indx);
        % locate event files based on task name
        switch conds(i).task
            case 'ampb'
                currSes = regexprep(currBold, '.+_(ses-\d+b?)_.+', '$1');
                currRun = regexprep(currBold, '.+_(run-\d)_.+', '$1');
                currEvents = cellfun(@(s,r) fullfile(paths.raw, s, 'func', ...
                    sprintf('%s_%s_task-%s_%s_events.tsv', ...
                    participant, s, conds(i).task, r)), currSes, currRun, ...
                    'UniformOutput', false);
            otherwise % mtlocal, ptlocal
                currSes = regexprep(currBold, '.+_(ses-\d+b?)_.+', '$1');
                currRun = regexprep(currBold, '.+_(run-\d)_.+', '$1');
                currEvents = cellfun(@(s,r) fullfile(paths.data, 'analysis', 'fMRI_data', participant, 'eventfiles', ...
                    sprintf('%s_%s_task-%s_%s_events.tsv', ...
                    participant, s, conds(i).task, r)), currSes, currRun, ...
                    'UniformOutput', false);
                %currEvents = fullfile(paths.data, 'analysis', 'fMRI_data', participant, 'eventfiles', currEvents);
                % currEvents = repmat({currEvents}, size(currBold));
        end
        [data,t] = create_data_matrix(currBold, TR);
        [design,stimdur,condnames] = create_design_matrix(currEvents, t);
        %%% load GLMsingle results (beta weights) and single-trial design matrix
        clear results designSINGLE; % clear previous
        switch glmMode
            case 'surface'
                outDir = sprintf('%s/%s', conds(i).task, conds(i).hemi);
            case 'volume'
                outDir = sprintf('task-%s', conds(i).task);
        end
        paths.glm = fullfile(paths.save, outDir);
        results = load(fullfile(paths.glm, 'TYPED_FITHRF_GLMDENOISE_RR.mat'));
        load(fullfile(paths.glm, 'DESIGNINFO.mat'), 'designSINGLE');

        %%% write surface maps (.mgz), task-dependent
        switch conds(i).task
            case {'mtlocal', 'ptlocal'} % collapse across runs and perform constrast

                clear modelmd dataMat R2 HRFindex; % clear previous
                matSize = size(results.modelmd); nv = prod(matSize(1:3)); % save original size

                %%% unscale beta weights that were saved as percent signal change
                %%% for contrasts: beta_scaled = <beta> / mean * 100
                modelmd = squeeze(results.modelmd) ./ 100; % remove singleton dimensions and remove percent
                modelmd = bsxfun(@times, modelmd, results.meanvol); % amplitude %@times = elementwise multiplication
                modelmd = reshape(modelmd, nv, matSize(end)); % flatten

                % reshape data for easier matrix calculations
                dataMat = data; % initialize with copy of the data
                for i2 = 1:length(data); dataMat{i2} = reshape(data{i2}, nv, []); end
                HRFindex = reshape(results.HRFindex, nv, []);
                R2 = reshape(results.R2, nv, []);

                %%% assign condition names to single-trial events
                indx = cellfun(@find_condition_index, design, 'UniformOutput', false);
                betaConditions = condnames(cat(1, indx{:}));
                nt = size(dataMat{1}, 2); % number of time points

                %%% create constrast vector
                c = zeros(length(betaConditions), 1); % initialize
                contrastStr = contrasts.(conds(i).task); % task-based constrast
                contrastStr = strsplit(regexprep(contrastStr, '\s', ''), ':');
                contrastStr = regexp(contrastStr, '(?<name>\w+)=(?<value>[\-\w]+)', 'names');
                contrastStr = cat(1, contrastStr{:}); % uncell structures
                for i2 = 1:length(contrastStr) % for each contrast
                    indx = strcmp(betaConditions, contrastStr(i2).name);
                    c(indx) = str2double(contrastStr(i2).value);
                end

                %%% compute design matrix for each hrf in hrf library
                hrfs = num2cell(getcanonicalhrflibrary(stimdur, TR), 2); % hrf library
                d = sum(cat(3, designSINGLE{:}), 3); % collapse single-trial design matrix
                X = cellfun(@(x) convn(d, x(:)), hrfs, 'UniformOutput', false); %convolve
                X = cellfun(@(x) TR .* x(1:nt,:), X, 'UniformOutput', false); %get the integral of the hrf

                %%% compute variance in the residuals for each vertex
                q = 1 - (R2 ./ 100); % variance NOT explained, proportion
                e = q .* var(cat(3, dataMat{:}), [], 2:3, 'omitnan'); % variance of residuals for each vertex

                %%% compute numerator (contrast-weighted beta weights) and
                %%% demoninator (contrast-weighted design matrix) for each hrf
                w = c' * modelmd'; % weighted contrast
                terror = cellfun(@(x) c' * pinv(x' * x) * c, X);

                %%% calculate t-statistic and p-value for each vertex
                t = w(:) ./ sqrt(e .* terror(HRFindex));
                p = 2 .* tcdf(-abs(t), abs(length(c) - nt));

                %%% save constrast t-statistic map
                clear mri; mri.vol = reshape(t, matSize(1:3)); % assign to new mri
                contrastDesc = strjoin(lower({contrastStr.name}), 'X');

                %%% save constrast t-statistics map
                switch glmMode
                    case 'surface'
                        saveName = sprintf( ...
                            '%s_task-%s_%s_space-fsnative_desc-%s_tstat.mgz', ...
                            participant, conds(i).task, conds(i).hemi, contrastDesc);
                    case 'volume'
                        saveName = sprintf( ...
                            '%s_task-%s_space-T1w_desc-%s_tstat.nii.gz', ...
                            participant, conds(i).task, contrastDesc);
                end
                if ~isfile(fullfile(paths.save, saveName))
                    MRIwrite(mri, fullfile(paths.save, saveName));
                    fprintf('Saved: %s\n', saveName);
                end

                %%% save constrast p-value map
                clear mri; mri.vol = reshape(p, matSize(1:3)); % assign to new mri
                saveName = regexprep(saveName, '_tstat', '_pval');
                if ~isfile(fullfile(paths.save, saveName))
                    MRIwrite(mri, fullfile(paths.save, saveName));
                    fprintf('Saved: %s\n', saveName);
                end
        end
    end
end