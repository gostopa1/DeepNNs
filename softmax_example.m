clear

% Uncomment the next two lines if you wish to run a classification tutorial
make_example_classification_data
model.type='classification';


% Model initialization 
example_model_initialization

% Model training
model2=model_train(model);
model.layers(end).activation='softmaxact2';
model3=model_train(model);
% Model evaluation
%[ ~,out]=forwardpassing(model,x); % The first output argument corresponds to the updated model but it is not needed here

%% Visualize the network
figure(2)
clf
show_network(model)

% Visualize the error over training epochs - any scattered behavior depends
% to the minibatch size 
figure(3)
clf
hold on
plot(model2.error)
plot(model3.error)
xlabel('Epochs')
ylabel('Error')
legend({'1','s*(1-s)'})
print('softmax_der.png','-dpng','-r300')

