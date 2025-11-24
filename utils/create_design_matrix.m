function [design,stimdur,allConditions] = create_design_matrix(eventFiles, t)
    %%% figure out all conditions across event files
    allConditions = cellfun(@(x) unique(cellstr(tdfread(x).trial_type)), ...
        eventFiles, 'UniformOutput', false);
    allConditions = setdiff(unique(cat(1, allConditions{:})), {'blank'});
    
    design = cell(1, length(eventFiles)); % initialize
    for i = 1:length(eventFiles) % for each event files
        %%% read events files and extract trial conditions
        events = tdfread(eventFiles{i}); % read events file
        conditions = cellstr(events.trial_type);

        %%% create design matrix per run
        X = zeros(length(t), length(allConditions)); stimdur = NaN; % initialize
        for i2 = 1:length(conditions) % for each condition
            k = strcmp(allConditions, conditions{i2}); 
            if any(k) % if task condition
                if isnan(stimdur); stimdur = events.duration(i2); end % collect stimdur
                tIndx = t >= events.onset(i2) & ...
                    t < (events.onset(i2) + events.duration(i2));
                indx = find(tIndx, 1); % only select the onset TR
                X(indx,k) = 1; % assign condition to design matrix
            end
        end

        %%% assign current run's design matrix into cell
        design{i} = X; 
    end
end
