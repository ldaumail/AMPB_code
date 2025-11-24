function combos = combvec_conditions(S)
% COMBVEC_CONDITIONS Generate all combinations of condition fields in a struct.
%
%   combos = combvec_conditions(S)
%
%   Input:
%       S : a struct whose fields each contain a cell array of strings
%           Example:
%               S.hemi = {'hemi-L','hemi-R'};
%               S.task = {'task1','task2','task3'};
%
%   Output:
%       combos : struct array with all combinations
%
%   Example:
%       S.hemi = {'L', 'R'};
%       S.task = {'A','B','C'};
%       combos = combvec_conditions(S)
%
%       → 6 combinations:
%          combos(1).hemi = 'L', combos(1).task = 'A'
%          combos(2).hemi = 'L', combos(2).task = 'B'
%          combos(3).hemi = 'L', combos(3).task = 'C'
%          combos(4).hemi = 'R', combos(4).task = 'A'
%          combos(5).hemi = 'R', combos(5).task = 'B'
%          combos(6).hemi = 'R', combos(6).task = 'C'

    fields = fieldnames(S);
    nFields = numel(fields);

    % Convert struct fields to a cell array of cell arrays
    lists = cell(1, nFields);
    for i = 1:nFields
        lists{i} = S.(fields{i});
    end

    % Compute sizes
    sizes = cellfun(@numel, lists);

    % Number of combinations
    total = prod(sizes);

    % Pre-allocate struct array
    combos = repmat(cell2struct(cell(nFields,1), fields, 1), total, 1);

    % Generate indices grid
    idx = cell(1,nFields);
    [idx{:}] = ndgrid(lists{:});

    % Fill combinations
    for k = 1:total
        for f = 1:nFields
            combos(k).(fields{f}) = idx{f}{k};
        end
    end
end
