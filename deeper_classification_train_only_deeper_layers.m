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
test_labs=xor(test_data(:,1)<0.5,test_data(:,2)<0.5); test_labs(:,2) = 1 - test_labs(:,1);
%x=x/max(x(:)); test_data=test_data/max(abs(test_data(:)));

%% Model Initialization
clear model

model.N=N;
model.x=x;
model.y=y;
layers=[5 10];
model.batchsize=500
noins=size(x,2);
noouts=size(y,2);
model.fe_update=100000

layers=[noins layers noouts];
lr=0.05; activation='tanhact';
%lr=0.02; activation='relu';
%lr=0.05; activation='logsi';

model.layersizes=[layers];
model.layersizesinitial=model.layersizes;

model.target=y;
model.epochs=1000;
model.update=100;
model.l2=1;
model.l1=1;
model.fe_update=10000;
model.fe_update=10000;
model.errofun='quadratic_cost';
model.errofun='cross_entropy_cost';
for layeri=1:(length(layers)-1)
    
    model.layers(layeri).lr=lr;
    model.layers(layeri).blr=lr;
    model.layers(layeri).Ws=[layers(layeri) layers(layeri+1)];
    %model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1))-0.5)/10;
    model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1)))*sqrt(2/(model.layersizes(layeri)+model.layersizes(layeri+1))); % Xavier initialization
    %model.layers(layeri).B=(randn(layers(layeri+1),1)-0.5)/10;
    model.layers(layeri).B=(zeros(layers(layeri+1),1))/10;
    model.layers(layeri).activation=activation;
    model.layers(layeri).inds=1:model.layersizes(layeri); % To keep track of which nodes are removed etc
end

%model.layers(layeri).lr=lr; model.layers(layeri).activation='softmaxact';
model.layers(layeri).lr=lr; model.layers(layeri).activation='softmaxact';
%% Model training


clear error
tic 
model=model_train_fast(model);
display(['Training all layers took ' sprintf('%2.2f',toc) ' seconds'])
figure(1)
clf
subplot(4,1,[1 2])
show_network(model)
%save_figure
%% Visual evaluation

model.test=0;
[~,out_test]=forwardpassing(model,[test_data]);
factor=15;

subplot(4,1,3)
hold on

for pointi=1:size(test_data,1)
    mat(((x_test(pointi)+1)*10)-9,((y_test(pointi)+1)*10)-9)=out_test(pointi,1);
end
contour(mat,[0.25 0.5 0.75],'LineWidth',3)
for pointi=1:size(test_data,1)
    if out_test(pointi,1)>0
        dotcol='k';
    else
        dotcol='r';
    end
    m=plot(((x_test(pointi)+1)*10)-9,((y_test(pointi)+1)*10)-9,[dotcol '.']);
    set(m,'MarkerSize',abs(out_test(pointi,1))*factor+5)
end
set(gca,'XTick',[1 11],'XTickLabel',[0 1]);
set(gca,'YTick',[1 11],'YTickLabel',[0 1]);

%axis off
%box off
subplot(4,1,4)
plot(model.error)
xlabel('Epoch')

ylabel('Error')

%% Numerical evaluation (i.e. classification accuracy, etc.)
[~,out_test]=forwardpassing_nolr(model,test_data);

display([sprintf('Performance : %3.2f%%',get_perf(out_test,test_labs))])

%%

model.layers(1).lr=0;
model.layers(2).lr=0;
tic

model=model_train_fast(model);
display(['Training only layer 3 took '  sprintf('%2.2f',toc) ' seconds'])


figure(2)
clf

subplot(4,1,[1 2])
show_network(model)

subplot(4,1,4)
plot(model.error)

[~,out_test]=forwardpassing(model,test_data);

display([sprintf('Performance : %3.2f%%',get_perf(out_test,test_labs))])
