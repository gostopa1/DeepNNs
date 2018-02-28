%% This script demonstrates the effect of different optimization algorithms thay may be used. 
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
    
    model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1)))/sqrt(model.layersizes(layeri));
    
    model.layers(layeri).mdedw=(zeros(layers(layeri),layers(layeri+1)))/sqrt(model.layersizes(layeri)); % This is necessary when momentum is implemented, since we always need the previous gradient.
    model.layers(layeri).mdedw_sq=(zeros(layers(layeri),layers(layeri+1)))/sqrt(model.layersizes(layeri)); % This is necessary for the RMSprop
    model.layers(layeri).momentum=(zeros(layers(layeri),layers(layeri+1)))/sqrt(model.layersizes(layeri)); % This is necessary for the RMSprop with momentum
    
    %model.layers(layeri).B=(randn(layers(layeri+1),1)-0.5)/10;
    model.layers(layeri).B=(zeros(layers(layeri+1),1))/1;
    model.layers(layeri).activation=activation;
    model.layers(layeri).inds=1:model.layersizes(layeri); % To keep track of which nodes are removed etc
    model.layers(layeri).g=0.99; % If this is set to zero then it is identical to simple gradient descent
end
%model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1))-0.5)/1;
%model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1)))/sqrt(layers(layeri));
%model.layers(layeri).lr=lr/0.1; model.layers(layeri).activation='tanhact';
model.layers(layeri).lr=lr/1; model.layers(layeri).activation='softmaxact';


%model.layers(layeri).lr=lr/10; model.layers(layeri).activation='linact';
%model.layers(layeri).lr=lr/10; model.layers(layeri).activation='tanhact';

%% Model training
display('Training with SGD')

model_simple=model_train_fast(model);
display(' ')
display('Training with Variation')

model_momentum=model_train_fast_momentum(model);
%%
figure(1)
clf
hold  on
plot(model_simple.error,'k','LineWidth',2)
plot(model_momentum.error,'r')
hold off
legend('Simple','Algorithm Variation')
xlabel('Epochs')
ylabel('Error')
return
show_network(model)
%save_figure
%% Visual evaluation

model.test=0;
[model,out_test]=forwardpassing(model,[test_data]);
factor=15;
figure(1)
clf
subplot(4,1,[1 2])
show_network(model)

subplot(4,1,3)
hold on

for pointi=1:size(test_data,1)
    mat(((x_test(pointi)+1)*10)-9,((y_test(pointi)+1)*10)-9)=out_test(pointi,1);
end
contour(mat,[0.25 0.5 0.75],'LineWidth',3)
test_labs=xor(x_test<0.5,y_test<0.5);
for pointi=1:size(test_data,1)
    if out_test(pointi,1)>0
        dotcol='k';
    else
        dotcol='r';
    end
    m=plot(((x_test(pointi)+1)*10)-9,((y_test(pointi)+1)*10)-9,[dotcol '.']);
    %text(((x_test(pointi)+1)*10)-9,((y_test(pointi)+1)*10)-9,num2str(test_labs(pointi)));
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
[model,out]=forwardpassing(model,x);

%[~,indpre]=max(out(:,:,end)');
[~,indpre]=max(out(:,:)');

%indpre-1
[~,indtarget]=max(y');
perf=(sum(indpre==indtarget)/length(indpre))*100;
display([sprintf('Performance : %3.2f%%',perf)])
