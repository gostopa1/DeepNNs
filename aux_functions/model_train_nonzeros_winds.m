function model=model_train_winds(model)
%% This is an altered version of the model_train function, aiming to speed up training.
%% To speed up training three approaches:
%%      - Only weights that are non-zero are updated
%%      - Only layers that have non-zero learning rate are updated (obvious but now the caculations are ommited)
%%      - Only one batch is used per epoch
%%      - Output and gradients are not re-calculated if learning rate is 0
for layeri=1:(length(model.layers))
    model.layers(layeri).nonzeroinds=find(model.layers(layeri).W~=0);
end
model.epoch=1;

[model,out(:,:,model.epoch)]=forwardpassing(model,model.x);

model.fnzl=find_first_non_zero_layer(model);
if model.fnzl==length(model.layers)
display('All learning rates are zero. The model will be identical! No point continuing!');
return;
end
for epoch=1:model.epochs
    model.epoch=epoch;
    if mod(epoch,model.update)==0
        outemp=out(:,:,epoch-1);
        %display(['Epoch: ' num2str(epoch)])
        if sum(isfield(model,{'x_test','y_test'}))==2
            epochtemp=model.epoch; model.epoch=0;
            [~,out_test]=forwardpassing(model,model.x_test(:,model.layers(1).inds)); get_perf(out_test,model.y_test);
            model.epoch=epochtemp;
            y_test_str=[' - Test: ' sprintf('%2.2f', get_perf(out_test,model.y_test))];
        else
            y_test_str='';
        end
        display(['Epoch: ' num2str(epoch) ' - Training: ' sprintf('%2.2f',get_perf(outemp,model.target)) y_test_str]);
    end
    if mod(epoch,model.fe_update)==0
        model=feature_elimination(model);
        show_network
        
        for layeri=1:(length(model.layers))
            model.layers(layeri).nonzeroinds=find(model.layers(layeri).W~=0);
        end
    end
    %%Forward passing
    model=vectorize_all_weights(model);
    
    [model,out(:,:,epoch)]=forwardpassing_nolr(model,model.x);
    [model.error(epoch),dedout]=feval(model.errofun,model);
    
    
    randomized_sample_indices=randperm(model.N); % Randomize the sample to randomly select samples for mini-batch
    randomized_sample_indices=[randomized_sample_indices randomized_sample_indices];
    for batchi=1:model.batchsize:model.N
        %for batchi=1
        for layeri=(length(model.layers)):-1:model.fnzl
            clear dedw dedb
            ins=model.layers(layeri).Ws(1);
            outs=model.layers(layeri).Ws(2);
            batchinds=randomized_sample_indices(batchi:(model.batchsize+batchi-1));
            if layeri==length(model.layers)
                % For the last layer the gradient is calculated as
                % dE/dw = dE/dout * dout/dnet * dnet/dw
                dnetdw=repmat(model.layers(layeri).X(batchinds,:),1,1,outs); % dnet/dw
                
                dedb=(dedout(batchinds,:).*model.layers(layeri).doutdnet(batchinds,:))'; % dE/db = dE/dout * dout/dnet
                dedw=permute(repmat(dedb,1,1,ins),[2 3 1]).*dnetdw; % dE/dw = dE/dout * dout/dnet * dnet/d
            else
                %dedb=permute(sum(permute(repmat(model.layers(layeri+1).grad,1,1,outs),[2 3 1]).*permute(repmat(model.layers(layeri+1).W,1,1,model.batchsize),[3 1 2]),3).*model.layers(layeri).doutdnet(batchinds,:),[2 1]);
                dedb=permute(mean(permute(repmat(model.layers(layeri+1).grad,1,1,outs),[2 3 1]).*permute(repmat(model.layers(layeri+1).W,1,1,model.batchsize),[3 1 2]),3).*model.layers(layeri).doutdnet(batchinds,:),[2 1]);
                dedw=permute(repmat(dedb,1,1,ins),[2 3 1]).*repmat(model.layers(layeri).X(batchinds,:),1,1,outs);
                
            end
            %layeri
            %size(dedb)
            model.layers(layeri).grad=dedb;
            if model.layers(layeri).lr~=0
                
                l1part=-model.layers(layeri).lr*(model.l1/model.batchsize).*sign(model.layers(layeri).W);
                l2part=-model.layers(layeri).lr*(model.l2/model.batchsize).*model.layers(layeri).W;
                regularization_term=model.layers(layeri).W+l1part+l2part;
                mdedw=permute(mean(dedw,1),[2 3 1]);
                %temp=regularization_term(model.layers(layeri).nonzeroinds)-model.layers(layeri).lr(model.layers(layeri).nonzeroinds).*mdedw(model.layers(layeri).nonzeroinds);
                %if sum(isnan(temp))>0
                %    display('About to explode')
                %    pause
                %end
                
                %model.layers(layeri).W(model.layers(layeri).nonzeroinds)=regularization_term(model.layers(layeri).nonzeroinds)-model.layers(layeri).lr(model.layers(layeri).nonzeroinds).*mdedw(model.layers(layeri).nonzeroinds);
                model.layers(layeri).W(model.layers(layeri).nonzeroinds)=regularization_term(model.layers(layeri).nonzeroinds)-model.layers(layeri).lr.*mdedw(model.layers(layeri).nonzeroinds);
                model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr.*mean(dedb,2);
                
            end
        end
    end
    
end

end