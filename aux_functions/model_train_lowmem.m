function [model,best_model]=model_train_lowmem(model)
%% This is an altered version of the model_train_fast function, aiming reduce memory requirements for large datasets.
%% To reduce memory, gradients for weights are calculated separately for each input node, and instead of storing all the values for each point, the averages are stored.


for layeri=1:(length(model.layers))
    model.layers(layeri).nonzeroinds=find(model.layers(layeri).W~=0);
end
model.epoch=1;

[model,out2(:,:,model.epoch)]=forwardpassing(model,model.x(randperm(model.N,model.batchsize),:));

model.fnzl=find_first_non_zero_layer(model);
if (length(model.layers)>1) && (model.fnzl==length(model.layers))
    display('All learning rates are zero. The model will be identical! No point continuing!');
    return;
end
best_model=model; % In the extreme case that the best_model is not assigned at all.
for epoch=1:model.epochs
    batchinds=randperm(model.N,model.batchsize);
    model.target=model.y(batchinds,:);
    model.epoch=epoch;
    if mod(epoch,model.update)==0
        % Now it is time to show an update of the network.
        if sum(rem(model.target,1)==model.target)==length(model.target)
            display(['Epoch: ' num2str(epoch) ' - Training: ' sprintf('%2.2f',model.error(epoch-1))]);
        else
            % If there is a test set, calculate the accuracy for it
            if sum(isfield(model,{'x_test','y_test'}))==2
                epochtemp=model.epoch; model.epoch=0;
                [~,out_test]=forwardpassing(model,model.x_test(:,model.layers(1).inds)); get_perf(out_test,model.y_test);
                model.epoch=epochtemp;
                y_test_str=[' - Test: ' sprintf('%2.2f', get_perf(out_test,model.y_test))];
            else
                y_test_str='';
            end
            [~,out_train]=forwardpassing(model,model.x);
            display(['Epoch: ' num2str(epoch) ' - Training: ' sprintf('%2.2f',get_perf(out_train,model.y)) y_test_str]);
        end
    end
    % Should I perform feature elimination
    if mod(epoch,model.fe_update)==0
        model=feature_elimination(model);
    end
    
    %%Forward passing
    model=vectorize_all_weights(model);
    
    %[model,out(:,:,epoch)]=forwardpassing_nolr(model,model.x(batchinds,:));
    [model,out(:,:,epoch)]=forwardpassing_nolr(model,model.x(batchinds,:));
    
    if strcmp(model.errofun,'cross_entropy_cost')
        [model.error(epoch),dedout]=cross_entropy_cost(model);
    elseif strcmp(model.errofun,'quadratic_cost')
        [model.error(epoch),dedout]=quadratic_cost(model);
    else
        [model.error(epoch),dedout]=feval(model.errofun,model);
    end
    if epoch>1
        if model.error(epoch)==min(model.error)
            best_model=model;
        end
    end
    
    for layeri=(length(model.layers)):-1:model.fnzl
        clear mdedw dedw dedb
        ins=model.layers(layeri).Ws(1);
        outs=model.layers(layeri).Ws(2);
        
        if layeri==length(model.layers)
            % For the last layer the gradient is calculated as
            % dE/dw = dE/dout * dout/dnet * dnet/dw
            dnetdw=repmat(model.layers(layeri).X,1,1,outs); % dnet/dw
            
            %dedb=(dedout(batchinds,:).*model.layers(layeri).doutdnet(batchinds,:))'; % dE/db = dE/dout * dout/dnet
            dedb=(dedout.*model.layers(layeri).doutdnet)'; % dE/db = dE/dout * dout/dnet
            mdedw=permute(mean(permute(repmat(dedb,1,1,ins),[2 3 1]).*dnetdw,1),[2 3 1]); % dE/dw = dE/dout * dout/dnet * dnet/d
        else
            
            
            dedb=permute(mean(permute(repmat(model.layers(layeri+1).grad,1,1,outs),[2 3 1]).*permute(repmat(model.layers(layeri+1).W,1,1,model.batchsize),[3 1 2]),3).*model.layers(layeri).doutdnet,[2 1]);
            
            %dedw=permute(repmat(dedb,1,1,ins),[2 3 1]).*repmat(model.layers(layeri).X,1,1,outs);
            for wi=1:model.layersizes(layeri)
                mdedw(wi,:)=mean(dedb.*repmat(model.layers(layeri).X(:,wi),1,outs)',2);
            end
            
        end
        %layeri
        %size(dedb)
        model.layers(layeri).grad=dedb;
        
        
        l1part=-model.layers(layeri).lr*(model.l1/model.batchsize).*sign(model.layers(layeri).W);
        l2part=-model.layers(layeri).lr*(model.l2/model.batchsize).*model.layers(layeri).W;
        regularization_term=model.layers(layeri).W+l1part+l2part;
        %mdedw=permute(mean(dedw,1),[2 3 1]);
        
        %model.layers(layeri).W(model.layers(layeri).nonzeroinds)=regularization_term(model.layers(layeri).nonzeroinds)-model.layers(layeri).lr(model.layers(layeri).nonzeroinds).*mdedw(model.layers(layeri).nonzeroinds);
        model.layers(layeri).W(model.layers(layeri).nonzeroinds)=regularization_term(model.layers(layeri).nonzeroinds)-model.layers(layeri).lr.*mdedw(model.layers(layeri).nonzeroinds);
        model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr.*mean(dedb,2);
        
        
    end
    
    
end

end
