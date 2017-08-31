clear
addpath('./activation_functions')
addpath('./loss_functions')
N=100;
x=randn(N,1);
%x=x(abs(x)>1);
N=length(x);
y=x.^2;
%y=tanh(x)

%x=x/max(x(:));
%y=y/max(y(:));
x=zscore(x,[],1)
y=zscore(y,[],1)

%%
%% The next part is identical in regression and classification
clear model

layers=[5 5];
batchsize=20
noins=size(x,2);
noouts=size(y,2);

layers=[noins layers noouts];
lr=0.1; activation='tanhact';
%lr=0.1; activation='relu';
%lr=1; activation='logsi';

model.layersizes=[layers];
model.target=y;
for layeri=1:(length(layers)-1)
    model.layers(layeri).lr=lr;
    model.layers(layeri).blr=lr;
    model.layers(layeri).Ws=[layers(layeri) layers(layeri+1)]
    model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1))-0.5)/10;
    model.layers(layeri).B=(randn(layers(layeri+1),1)-0.5)/10;
            
    model.layers(layeri).activation=activation;
    
end
layeri
model.layers(layeri).lr=lr/10; model.layers(layeri).activation='linact';
%model.layers(layeri).lr=lr/10; model.layers(layeri).activation='tanhact';
%model.layers(layeri).lr=lr/100; model.layers(layeri).activation='softmaxact';

clear error
epochs=1000
for epoch=1:epochs
    display(['Epoch: ' num2str(epoch)])
    %%Forward passing
    [model,out(:,:,epoch)]=forwardpassing(model,x);
    
    [error(epoch),dedout]=quadratic_cost(model); % Using quadratic error function
    
    %[error(epoch),dedout]=cross_entropy_cost(model); % Using quadratic error function
    randomized_sample_indices=randperm(N);
    for batchi=1:batchsize:N
        layeri=length(model.layers);
        %%batchinds=randperm(N,batchsize);
        %batchi
        batchinds=randomized_sample_indices(batchi:(batchsize+batchi-1));
        
        clear dedw dedb
        
        ins=model.layers(layeri).Ws(1);
        outs=model.layers(layeri).Ws(2);
                
        dnetdw=permute(repmat(model.layers(layeri).X(batchinds,:),1,1,outs),[2 3 1]);
        dedb=(dedout(batchinds,:).*model.layers(layeri).doutdnet(batchinds,:))';
        dedw=permute(repmat(dedb,1,1,ins),[3 1 2]).*dnetdw;
        
        %model.layers(layeri).grad=dedout.*model.layers(layeri).doutdnet;
        model.layers(layeri).grad=dedb';
        
        %model.layers(layeri).W=model.layers(layeri).W-model.layers(layeri).lr*sum(dedw,3);
        %model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr*sum(dedb,2);
        
        model.layers(layeri).W=model.layers(layeri).W-model.layers(layeri).lr*mean(dedw,3);
        model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr*mean(dedb,2);
        
        for layeri=(length(model.layers)-1):-1:1
            clear dedw dedb
            ins=model.layers(layeri).Ws(1);
            outs=model.layers(layeri).Ws(2);
            
            %model.layers(layeri).grad=squeeze(sum(permute(repmat(model.layers(layeri+1).grad,1,1,outs),[3 2 1]).*repmat(model.layers(layeri+1).W,1,1,batchsize),2))'.*model.layers(layeri).doutdnet(batchinds,:);
            model.layers(layeri).grad=permute(sum(permute(repmat(model.layers(layeri+1).grad,1,1,outs),[3 2 1]).*repmat(model.layers(layeri+1).W,1,1,batchsize),2),[1 3 2])'.*model.layers(layeri).doutdnet(batchinds,:);
            
            dedb=model.layers(layeri).grad';
            dedw=permute(repmat(dedb,1,1,ins),[3 1 2]).*permute(repmat(model.layers(layeri).X(batchinds,:),1,1,outs),[2 3 1]);
                        
            %model.layers(layeri).W=model.layers(layeri).W-model.layers(layeri).lr*sum(dedw,3);
            %model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr*sum(dedb,2);
            
            model.layers(layeri).W=model.layers(layeri).W-model.layers(layeri).lr*mean(dedw,3);
            model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr*mean(dedb,2);        
        end
    end
    
    if mod(epoch,100)==1
        %model.layers(end).W
        %model.layers(end).grad
        %pause
        
        show_network
        drawnow
    end
end
%%
%show_network
%print('a.png','-dpng','-r1000')

%[model,out]=forwardpassing(model,x)


subplot(4,1,3)
hold on
h=plot(x,y,'b.')
ms=10
set(h,'MarkerSize',ms)
%plot(x,out,'r.')
for epochi=1:size(out,3)
    h=plot(x,squeeze(out(:,:,epochi)),'.','Color',[[1 0 0] *((1-(epochi/epochs)))]);
    set(h,'MarkerSize',ms)
end
h=plot(x,squeeze(out(:,:,epoch)),'k.')
set(h,'MarkerSize',ms)


subplot(4,1,4)
plot(error)