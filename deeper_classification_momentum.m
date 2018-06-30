%% This script demonstrates the effect of different optimization algorithms that are available in the toolbox
% Check out the model_train_fast_momentum function to see multiple
% variations

%% Creating dataset

clear
addpath('./aux_functions')
addpath('./activation_functions')
addpath('./loss_functions')

x = [1 1; 1 0; 0 1; 0 0];
y(:,2) = [1; 0; 0; 1];
y(:,1) = [0; 1; 1; 0];
N=size(x,1);

N=1000; x=rand(N,2); y=xor(x(:,1)<0.5,x(:,2)<0.5); y(:,2) = 1 - y(:,1);

x=zscore(x,[],1);

%y=zscore(y,[],1)

[x_test,y_test]=meshgrid(0:0.1:1,0:0.1:1);

%test_data=[x_test(: ) y_test(:)];
test_data=zscore([x_test(: ) y_test(:)],1);

%x=x/max(x(:)); test_data=test_data/max(abs(test_data(:)));

%% Model Initialization
clear model

layers=[10];
model.batchsize=500;
model.type='classification';
model.N=N;
model.x=x;
model.y=y;
model.fe_update=100000;
noins=size(x,2);
noouts=size(y,2);

layers=[noins layers noouts];
lr=0.1; activation='tanhact';
%lr=0.01; activation='relu';
%lr=0.1; activation='logsi';
model.l2=0;
model.l1=0;
model.layersizes=[layers];
model.layersizesinitial=model.layersizes;
model.target=y;
model.epochs=1000;
model.update=100;
model.errofun='quadratic_cost';
model.errofun='cross_entropy_cost';
for layeri=1:(length(layers)-1)
    model.layers(layeri).lr=lr;
    model.layers(layeri).blr=lr;
    model.layers(layeri).Ws=[layers(layeri) layers(layeri+1)]
    
    model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1)))*sqrt(2/(model.layersizes(layeri)+model.layersizes(layeri+1)));
    
    %model.layers(layeri).B=(randn(layers(layeri+1),1)-0.5)/10;
    model.layers(layeri).B=(zeros(layers(layeri+1),1))/1;
    model.layers(layeri).activation=activation;
    model.layers(layeri).inds=1:model.layersizes(layeri); % To keep track of which nodes are removed etc
    
end
%model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1))-0.5)/1;
%model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1)))/sqrt(layers(layeri));
%model.layers(layeri).lr=lr/0.1; model.layers(layeri).activation='tanhact';
model.layers(layeri).lr=lr/1; model.layers(layeri).activation='softmaxact';


%model.layers(layeri).lr=lr/10; model.layers(layeri).activation='linact';
%model.layers(layeri).lr=lr/10; model.layers(layeri).activation='tanhact';

%% Model training
figure(1)
clf
hold on
optimizers={'SGD','SGD_m','RMSprop','RMSprop_m','ADAM'};
for optimizer=optimizers
    display(optimizer{1});
    model.optimizer=optimizer{1};
    display(['Training with ' model.optimizer])
    model1=model_train(model);
    
    plot(model1.error,'LineWidth',2)
end

legend(optimizers)
hold off
xlabel('Epochs')
ylabel('Error')
title('Model error convergence comparison')
