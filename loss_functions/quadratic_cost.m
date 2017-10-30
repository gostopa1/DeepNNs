function [error,dedout]=quadratic_cost(model)
%error=sum(mean(0.5*(model.layers(end).out-model.target).^2));



error=sum(mean(0.5*(model.layers(end).out-model.target).^2))+model.l2/(2*model.batchsize)*sum(model.allweights(:).^2)+model.l1/(2*model.batchsize)*sum(model.allweights(:));

dedout=(model.layers(end).out-model.target);

end