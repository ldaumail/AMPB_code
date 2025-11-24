
glmMode = 'surface';

participants = readlines('./study2_subjects_updated.txt');
for i =1:length(participants)
    participant = participants(i);
    
    run_GLMsingle(participant, glmMode)
end