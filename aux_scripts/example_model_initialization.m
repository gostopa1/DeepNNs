
if ~isfield(model,'type')
    model.type='regression';
end

layers=[5 5];
model.batchsize=200; % Number of samples to use in each epoch
model.N=N;
model.x=x;
model.y=y;
model.fe_update=100000;

noins=size(x,2);
noouts=size(y,2);

layers=[noins layers noouts];
lr=0.001; activation='tanhact'; % Hyperbolic function as activation function 
%lr=0.001; activation='relu'; % Using REctified Linear Unit as activation function
%lr=0.1; activation='logsi'; % Logistic function as activation function
model.l2=0;
model.l1=0;
model.layersizes=[layers];
model.layersizesinitial=model.layersizes;
model.target=y;
model.epochs=10000; % How many training epochs to perform
model.update=100; % How often to show update of the network
%model.errofun='quadratic_cost';
model.errofun='cross_entropy_cost'; % This error function is widely used for classification problems where the distribution of the output values is not gaussian (typically it is binomial)
for layeri=1:(length(layers)-1)
    model.layers(layeri).lr=lr;
    model.layers(layeri).blr=lr;
    model.layers(layeri).Ws=[layers(layeri) layers(layeri+1)];
    
    model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1)))*sqrt(2/(model.layersizes(layeri)+model.layersizes(layeri+1)));
    
    %model.layers(layeri).B=(randn(layers(layeri+1),1)-0.5)/10;
    model.layers(layeri).B=(zeros(layers(layeri+1),1))/1;
    model.layers(layeri).activation=activation;
    model.layers(layeri).inds=1:model.layersizes(layeri); % To keep track of which nodes are removed etc
    
end

if strcmp(model.type,'classification')
model.layers(layeri).activation='softmaxact'; % This is used for classification problems to normalize the output probabilities
else
    model.layers(layeri).activation='linact'; % This is used to avoid any limitation of the activation functions that might project values only to the positive side (ReLU [0,+inf]) or limit them between -1 and 1 (tanh) etc.
end

model.optimizer='ADAM';
%model.optimizer='SGD';
%model.optimizer='SGD_m';
%model.optimizer='RMSprop_m';