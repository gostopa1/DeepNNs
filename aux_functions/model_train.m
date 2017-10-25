function model=model_train(model,x,y)


for epoch=1:model.epochs
    if mod(epoch,model.update)==0
        display(['Epoch: ' num2str(epoch)])
    end
    %%Forward passing
    
    [model,out(:,:,epoch)]=forwardpassing(model,x);
    [model.error(epoch),dedout]=feval(model.errofun,model);
    
    randomized_sample_indices=randperm(model.N); % Randomize the sample to randomly select samples for mini-batch
    for batchi=1:model.batchsize:model.N
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
                
                model.layers(layeri).W=model.layers(layeri).W-model.layers(layeri).lr.*mean(dedw,3);
                model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr*mean(dedb,2);
            end
        end
    end
    
end

end