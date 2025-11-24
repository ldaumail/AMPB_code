function [data,t] = create_data_matrix(boldFiles, TR)
    data = cell(1, length(boldFiles)); % initialize
    for i = 1:length(boldFiles) % for each bold file
        [~,~,ext] = fileparts(boldFiles{i}); 
        if contains(ext, 'gii'); bold = gifti(boldFiles{i}).cdata; 
        else; bold = MRIread(boldFiles{i}).vol; % volume bold
        end
        data{i} = single(bold); % assign run to data cell
        if i == 1; t = ((1:size(bold, ndims(bold))) - 1) .* TR; end
    end
end
