function model=model_train(model,x,y)

[model,out(:,:,1)]=forwardpassing(model,model.x);

for epoch=1:model.epochs
    model.epoch=epoch;
    if mod(epoch,model.update)==0
        outemp=model.out(:,:,model.epoch-1);
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
    %%Forward passing
    %prev_model=model; % In case of NaN it is useful to see what was the previous model that led to this NaN values
    model=vectorize_all_weights(model);
    [model,model.out(:,:,epoch)]=forwardpassing(model,model.x);
    [model.error(epoch),dedout]=feval(model.errofun,model);
    
    randomized_sample_indices=randperm(model.N); % Randomize the sample to randomly select samples for mini-batch
    randomized_sample_indices=[randomized_sample_indices randomized_sample_indices]; % In case N is not a integer multiple of the the minibatch size, this avoids errors by adding the same vector twice
    for batchi=1:model.batchsize:model.N
        for layeri=(length(model.layers)):-1:find_first_non_zero_layer(model)
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
        show_network
        drawnow
    end
end

end