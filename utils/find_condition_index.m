function [cindx] = find_condition_index(design)
    [nt,cindx] = find(design); [~,indx] = sort(nt); cindx = cindx(indx); 
end