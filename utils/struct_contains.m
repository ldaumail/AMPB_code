function idx = struct_contains(boldfiles, conds)
% STRUCT_CONTAINS  Return indices of files matching all condition strings.
%
%   idx = struct_contains(boldfiles, conds)
%
%   Inputs:
%       boldfiles : cell array of file paths (strings)
%       conds     : struct whose fields contain required substrings
%
%   Output:
%       idx : indices of boldfiles that match all conditions
%

    fields = fieldnames(conds);
    nFields = numel(fields);

    matchFlags = false(size(boldfiles));

    for f = 1:numel(boldfiles)
        thisfile = boldfiles{f};
        match = true;

        for k = 1:nFields
            pattern = conds.(fields{k});
            if ~contains(thisfile, pattern)
                match = false;
                break
            end
        end

        matchFlags(f) = match;
    end

    idx = find(matchFlags);
end