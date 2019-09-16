%% Creating dataset

clear
addpath('./aux_functions')
addpath('./activation_functions')
addpath('./loss_functions')

x = [1 1; 1 0; 0 1; 0 0];
y(:,2) = [1; 0; 0; 1];
y(:,1) = [0; 1; 1; 0];
N=size(x,1);

N=1000; x=rand(N,10); y=xor(x(:,1)<0.5,x(:,2)<0.5); y(:,2) = 1 - y(:,1);

x=zscore(x,[],1);

%y=zscore(y,[],1)

[x_test,y_test]=meshgrid(0:0.1:1,0:0.1:1);

%test_data=[x_test(: ) y_test(:)];
test_data=zscore([x_test(: ) y_test(:)],1);

%x=x/max(x(:)); test_data=test_data/max(abs(test_data(:)));

%% Model Initialization
clear model
model.x=x;
layers=[5];
model.batchsize=200;
noins=size(x,2);
noouts=size(y,2);

layers=[noins layers noouts];
lr=0.1; activation='tanhact';
%lr=0.01; activation='relu';       
%lr=0.05; activation='logsi';

model.layersizes=[layers];
model.layersizesinitial=model.layersizes;
model.target=y;
model.epochs=1000;

model.update=100;
model.fe_update=400;
model.fe_thres=0.01;

model.l2=0;
model.l1=2;

model.update=50;
model.fe_update=250;
model.l2=1;
model.l1=0;


model.errofun='quadratic_cost';
model.errofun='cross_entropy_cost';
for layeri=1:(length(layers)-1)
    
    model.layers(layeri).lr=lr;
    model.layers(layeri).blr=lr;
    model.layers(layeri).Ws=[layers(layeri) layers(layeri+1)];
    
    model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1)))*sqrt(2/(model.layersizes(layeri)+model.layersizes(layeri+1)));
    
    model.layers(layeri).B=(zeros(layers(layeri+1),1))/10;
    model.layers(layeri).activation=activation;
    model.layers(layeri).inds=1:model.layersizes(layeri); % To keep track of which nodes are removed etc
end
model.fe_thres=[0.1 0 0 0]; % The feature elimination threshold can be either one per layer or a universal one
%model.fe_thres=0.0;

model.layers(layeri).lr=lr; model.layers(layeri).activation='softmaxact';
%% Model training

clear error

for epoch=1:model.epochs
    model.epoch=epoch;
    if mod(epoch,model.update)==0
        outemp=out(:,:,model.epoch-1);
        %display(['Epoch: ' num2str(model.epoch)])
        if sum(isfield(model,{'x_test','y_test'}))==2
            epochtemp=model.epoch; model.epoch=0;
            [~,out_test]=forwardpassing(model,[model.x_test]); get_perf(out_test,model.y_test);
            model.epoch=epochtemp;
            y_test_str=[' - Test: ' sprintf('%2.2f', get_perf(out_test,model.y_test))];
        else
            y_test_str='';
        end
        display(['Epoch: ' num2str(model.epoch) ' - Training: ' sprintf('%2.2f',get_perf(outemp,model.target)) y_test_str]);
        
    end
    if mod(epoch,model.fe_update)==0
        model=feature_elimination(model);
    end
    %%Forward passing
    prev_model=model;
    model=vectorize_all_weights(model);
    [model,out(:,:,epoch)]=forwardpassing(model,model.x);
    
    
    [error(epoch),dedout]=feval(model.errofun,model);
    
    randomized_sample_indices=randperm(N); % Randomize the sample to randomly select samples for mini-batch
    randomized_sample_indices=[randomized_sample_indices randomized_sample_indices]; % In case N is not a integer multiple of the the minibatch size, this avoids errors by adding the same vector twice
    for batchi=1:model.batchsize:N
        for layeri=(length(model.layers)):-1:1
            clear dedw dedb
            
            ins=model.layers(layeri).Ws(1);
            outs=model.layers(layeri).Ws(2);
            batchinds=randomized_sample_indices(batchi:(model.batchsize+batchi-1));
            if layeri==length(model.layers)
                % For the last layer the gradient is calculated as
                % dE/dw = dE/dout * dout/dnet * dnet/dw
                dnetdw=permute(repmat(model.layers(layeri).X(batchinds,:),1,1,outs),[2 3 1]); % dnet/dw
                
                dedb=(dedout(batchinds,:).*model.layers(layeri).doutdnet(batchinds,:))'; % dE/db = dE/dout * dout/dnet
                dedw=permute(repmat(dedb,1,1,ins),[3 1 2]).*dnetdw; % dE/dw = dE/dout * dout/dnet * dnet/d
            else
                dedb=(permute(sum(permute(repmat(model.layers(layeri+1).grad,1,1,outs),[3 2 1]).*repmat(model.layers(layeri+1).W,1,1,model.batchsize),2),[1 3 2])'.*model.layers(layeri).doutdnet(batchinds,:))';
                dedw=permute(repmat(dedb,1,1,ins),[3 1 2]).*permute(repmat(model.layers(layeri).X(batchinds,:),1,1,outs),[2 3 1]);
            end
            
            model.layers(layeri).grad=dedb';
            if model.layers(layeri).lr~=0
                
                l1part=-model.layers(layeri).lr*(model.l1/model.batchsize).*sign(model.layers(layeri).W);
                l2part=-model.layers(layeri).lr*(model.l2/model.batchsize).*model.layers(layeri).W;
                regularization_term=model.layers(layeri).W+l1part+l2part;
                
                temp=regularization_term-model.layers(layeri).lr*mean(dedw,3);
                if sum(isnan(temp))>0
                    display('About to explode')
                    pause
                end
                model.layers(layeri).W=regularization_term-model.layers(layeri).lr*mean(dedw,3);
                model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr*mean(dedb,2);
                
            end
        end
    end
    
    if mod(epoch,model.update)==0
        figure(1)
        clf
        subplot(4,1,[1 2])
        show_network(model)
        drawnow
    end
end
show_network(model)
model2=model;
%return
%save_figure
%% Visual evaluation

model.test=0;
try
    [model,out_test]=forwardpassing(model,[test_data]);
catch
    display('The remaining inputs do not match the test_data')
    return
end
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
plot(error)
xlabel('Epoch')

ylabel('Error')

%% Numerical evaluation (i.e. classification accuracy, etc.)

[~,indpre]=max(out(:,:,end)');

%indpre-1
[~,indtarget]=max(model.target');
perf=(sum(indpre==indtarget)/length(indpre))*100;
display([sprintf('Performance : %3.2f%%',perf)])
