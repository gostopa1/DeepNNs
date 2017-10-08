clear
addpath('./activation_functions')
addpath('./loss_functions')
N=500;
x=randn(N,1);
%x=x(abs(x)>1);
N=length(x);
y=x.^2;
%y=tanh(x)

x=x/max(x(:));
y=y/max(y(:));
%x=zscore(x,[],1)
%y=zscore(y,[],1)


%% Model initialization
clear model

layers=[5 5];
%batchsize=20
model.batchsize=50
noins=size(x,2);
noouts=size(y,2);

layers=[noins layers noouts];
lr=0.02; activation='tanhact';
%lr=0.1; activation='relu';
%lr=0.05; activation='logsi';

model.layersizes=[layers];
model.target=y;
model.epochs=500;
model.errofun='quadratic_cost';

model.update=100; % How often to provide feedback to the user
for layeri=1:(length(layers)-1)
    model.layers(layeri).lr=lr;
    model.layers(layeri).blr=lr;
    model.layers(layeri).Ws=[layers(layeri) layers(layeri+1)]
    model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1))-0.5)/10;
    model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1))-0.5)/1;
    model.layers(layeri).B=(randn(layers(layeri+1),1)-0.5)/1;
            
    model.layers(layeri).activation=activation;
    
end

%model.layers(layeri).lr=lr/10; model.layers(layeri).activation='linact';
model.layers(layeri).lr=lr/1; model.layers(layeri).activation='tanhact';

%% Model training

clear error

for epoch=1:model.epochs
    if mod(epoch,model.update)==0
        display(['Epoch: ' num2str(epoch)])
    end
    %%Forward passing
    
    [model,out(:,:,epoch)]=forwardpassing(model,x);
    [error(epoch),dedout]=feval(model.errofun,model);
    
    
    randomized_sample_indices=randperm(N); % Randomize the sample to randomly select samples for mini-batch
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
                
                model.layers(layeri).W=model.layers(layeri).W-model.layers(layeri).lr*mean(dedw,3);
                model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr*mean(dedb,2);
            end
        end
    end
    
    if mod(epoch,model.update)==0
        show_network
        drawnow
    end
end

%save_figure
%% Visual evaluation


subplot(4,1,3)
hold on
h=plot(x,y,'b.')
ms=10
set(h,'MarkerSize',ms)
%plot(x,out,'r.')
for epochi=1:size(out,3)
    h=plot(x,squeeze(out(:,:,epochi)),'.','Color',[[1 0 0] *((1-(epochi/model.epochs)))]);
    set(h,'MarkerSize',ms)
end
h=plot(x,squeeze(out(:,:,epoch)),'k.')
set(h,'MarkerSize',ms)


subplot(4,1,4)
plot(error)