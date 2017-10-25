function [model]=vectorize_all_weights(model)
model.allweights=[];
for layeri=1:length(model.layers)
    model.allweights=[model.allweights ; model.layers(layeri).W(:)];;
end
end