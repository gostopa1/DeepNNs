function [model,best_model]=model_train(model)
%% This is an altered version of the model_train function, aiming to speed up training.
%% To speed up training three approaches:
%%      - Only weights that are non-zero are updated
%%      - Only layers that have non-zero learning rate are updated (obvious but now the caculations are ommited)
%%      - Only one batch is used per epoch
%%      - Output and gradients are not re-calculated if learning rate is 0


for layeri=1:(length(model.layers))
    model.layers(layeri).nonzeroinds=find(model.layers(layeri).W~=0);
    [model.layers(layeri).insinds model.layers(layeri).outsinds]=ind2sub(model.layers(layeri).Ws,model.layers(layeri).nonzeroinds);
    
end
if ~isfield(model,'optimizer')
    display('No optimizer was selected')
    display('Setting SGD as the optimizer')
    model.optimizer='SGD';
end
for layeri=1:(length(model.layers))
    if strcmp(model.optimizer,'SGD')
    elseif strcmp(model.optimizer,'SGD_m')
        model.layers(layeri).mdedw=model.layers(layeri).W(model.layers(layeri).nonzeroinds)*0;
    elseif strcmp(model.optimizer,'RMSprop')
        model.layers(layeri).mdedw=model.layers(layeri).W(model.layers(layeri).nonzeroinds)*0;
        model.layers(layeri).mdedw_sq=model.layers(layeri).W(model.layers(layeri).nonzeroinds)*0; % This is necessary for the RMSprop
    elseif strcmp(model.optimizer,'RMSprop_m')
        model.layers(layeri).mdedw=model.layers(layeri).W(model.layers(layeri).nonzeroinds)*0;
        model.layers(layeri).mdedw_sq=model.layers(layeri).W(model.layers(layeri).nonzeroinds)*0;
    elseif strcmp(model.optimizer,'ADAM')
        model.layers(layeri).mdedw=model.layers(layeri).W(model.layers(layeri).nonzeroinds)*0;
        model.layers(layeri).m=model.layers(layeri).W(model.layers(layeri).nonzeroinds)*0;
        model.layers(layeri).u=model.layers(layeri).W(model.layers(layeri).nonzeroinds)*0;
    else
        display('No proper optimizer was chosen.')
        display('Setting SGD as the optimizer')
        model.optimizer='SGD';
    end
end

model.epoch=1;

[model,~]=forwardpassing(model,model.x(randperm(model.N,model.batchsize),:));

model.fnzl=find_first_non_zero_layer(model);
if (length(model.layers)>1) && (model.fnzl==length(model.layers))
    display('All learning rates are zero. The model will be identical! No point continuing!');
    return;
end

best_model=model;

for epoch=1:model.epochs
    batchinds=randperm(model.N,model.batchsize);
    model.target=model.y(batchinds,:);
    model.epoch=epoch;
    if mod(epoch,model.update)==0
        % Now it is time to show an update of the network.
        if strcmp(model.type,'regression')
            display(['Epoch: ' num2str(epoch) ' - Training: ' sprintf('%2.10f',model.error(epoch-1))]);
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
        if model.error(epoch)==min(model.error(:))
            best_model=model;
        end
    end
    for layeri=(length(model.layers)):-1:model.fnzl
        clear dedw dedb
        ins=model.layers(layeri).Ws(1);
        outs=model.layers(layeri).Ws(2);
        
        if layeri==length(model.layers)
            % For the last layer the gradient is calculated as
            % dE/dw = dE/dout * dout/dnet * dnet/dw
            dnetdw=repmat(model.layers(layeri).X,1,1,outs); % dnet/dw
            
            %dedb=(dedout(batchinds,:).*model.layers(layeri).doutdnet(batchinds,:))'; % dE/db = dE/dout * dout/dnet
            dedb=(dedout.*model.layers(layeri).doutdnet)'; % dE/db = dE/dout * dout/dnet
            dedw=permute(repmat(dedb,1,1,ins),[2 3 1]).*dnetdw; % dE/dw = dE/dout * dout/dnet * dnet/d
            mdedw=permute(mean(dedw,1),[2 3 1]);
            mdedw=mdedw(model.layers(layeri).nonzeroinds);
        else
            dedb=permute(mean(permute(repmat(model.layers(layeri+1).grad,1,1,outs),[2 3 1]).*permute(repmat(model.layers(layeri+1).W,1,1,model.batchsize),[3 1 2]),3).*model.layers(layeri).doutdnet,[2 1]);
            %dedw=permute(repmat(dedb,1,1,ins),[2 3 1]).*repmat(model.layers(layeri).X,1,1,outs);
            %mdedw=permute(mean(dedw,1),[2 3 1]);
            %mdedw=mdedw(model.layers(layeri).nonzeroinds);
            
            %mdedw=permute((repmat(mean(dedb,2),1,ins).*repmat(mean(model.layers(layeri).X,1),outs,1)),[2 1]);
            
            %mdedw=mdedw(model.layers(layeri).nonzeroinds);
            
            mdedw=mean(dedb(model.layers(layeri).outsinds,:).*(model.layers(layeri).X(:,model.layers(layeri).insinds)'),2);
        end
        %layeri
        %size(dedb)
        model.layers(layeri).grad=dedb;
        
        l1part=-model.layers(layeri).lr*(model.l1/model.batchsize).*sign(model.layers(layeri).W);
        l2part=-model.layers(layeri).lr*(model.l2/model.batchsize).*model.layers(layeri).W;
        regularization_term=model.layers(layeri).W+l1part+l2part;
        
        if model.layersizes(layeri)==1
            mdedw=mdedw';
        end
        
        if strcmp(model.optimizer,'SGD')
            %% SGD - No Momentum
            %mdedw=permute(mean(dedw,1),[2 3 1]);
            model.layers(layeri).mdedw=mdedw;
            
        elseif strcmp(model.optimizer,'SGD_m')
            %% SGD - with momentum
            g=0.9;
            %mdedw=permute(mean(dedw,1),[2 3 1]);
            model.layers(layeri).mdedw=(g)*mdedw+(1-g)*model.layers(layeri).mdedw;
            
        elseif strcmp(model.optimizer,'RMSprop')
            %% RMSprop - no momentum
            g=0.9;
            %mdedw=permute(mean(dedw,1),[2 3 1]);
            model.layers(layeri).mdedw_sq=(g)*model.layers(layeri).mdedw_sq+(1-g)*mdedw.^2;
            model.layers(layeri).mdedw=mdedw./(sqrt(model.layers(layeri).mdedw_sq)+10^(-8));
            
        elseif strcmp(model.optimizer,'RMSprop_m')
            %% RMSprop - with momentum
            g1=0.9;
            g=0.9;
            epsil=10^(-8);
            %mdedw=permute(mean(dedw,1),[2 3 1]);
            model.layers(layeri).mdedw_sq=(g1)*model.layers(layeri).mdedw_sq+(1-g1)*mdedw.^2;
            mdedw=mdedw./(sqrt(model.layers(layeri).mdedw_sq)+epsil);
            model.layers(layeri).mdedw=(g)*model.layers(layeri).mdedw+(1-g)*mdedw;
            
            %% ADAM
        elseif strcmp(model.optimizer,'ADAM')
            %if layeri==1
            %   display('koloky8ia') ;
            %end
            b1=0.9;
            b2=0.99;
            epsil=10^(-8);
            %model.layers(layeri).mdedw=permute(mean(dedw,1),[2 3 1]);
            %model.layers(layeri).mdedw=mdedw(model.layers(layeri).nonzeroinds);
            
            model.layers(layeri).mdedw=mdedw;
            model.layers(layeri).m=b1*model.layers(layeri).m+(1-b1)*model.layers(layeri).mdedw;
            model.layers(layeri).u=b2*model.layers(layeri).u+(1-b2)*model.layers(layeri).mdedw.^2;
            model.layers(layeri).mdedw=(model.layers(layeri).m./(1-b1^epoch))./(sqrt((model.layers(layeri).u./(1-b2^epoch)))+epsil);
            
        else
            display('Hey you have not chosen any optimizer!');
        end
        %% Now updating the weights
        
        if length(model.layers(layeri).lr)>1
            model.layers(layeri).W(model.layers(layeri).nonzeroinds)=regularization_term(model.layers(layeri).nonzeroinds)-model.layers(layeri).lr(model.layers(layeri).nonzeroinds).*model.layers(layeri).mdedw(model.layers(layeri).nonzeroinds);
        else
            model.layers(layeri).W(model.layers(layeri).nonzeroinds)=regularization_term(model.layers(layeri).nonzeroinds)-model.layers(layeri).lr*model.layers(layeri).mdedw;
        end
        model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr.*mean(dedb,2);
        
        
    end
    
    
end

end
