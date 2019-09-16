% This tutorial shows how to use the toolbox in a compact form that splits
% the process in three parts: data_generation, model initialization, model
% training and evaluation.
% In case of regression, you might need to run the script a few times to
% get good results. This depends on
%           1.data normalization 
%           2.weight initialization
%
% and it is left here for demonstration purposes.
clear
addpath(genpath('../DeepNNs'))
make_example_regression_data
model.type='regression';

% Uncomment the next two lines if you wish to run a classification tutorial
make_example_classification_data
model.type='classification';

% Model initialization 
example_model_initialization

% Model training
model=model_train(model);

% Model evaluation
[ ~,out]=forwardpassing(model,x); % The first output argument corresponds to the updated model but it is not needed here

%%
figure(1)
clf
hold
if strcmp(model.type,'regression')
    % Show how the original and the modeled data lie on the 2 dimensions
    plot(x,y,'.')
    plot(x,out,'.')
else
    
    for xi=1:size(x,1)
        h=plot(x(xi,1),x(xi,2),'.');
        set(h,'MarkerSize',out(xi,1)*10+5,'Color',round(out(xi,1)*[1 0 0])); % Marker size indicates probability to belong to class 1
    end
end

%% Visualize the network
figure(2)
clf
show_network(model)

% Visualize the error over training epochs - any scattered behavior depends
% to the minibatch size 
figure(3)
plot(model.error)
xlabel('Epochs')
ylabel('Error')

