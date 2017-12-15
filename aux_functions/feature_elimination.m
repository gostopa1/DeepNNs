function model=feature_elimination(model)

if ~isfield(model,'fe_thres')
    thres=0.1;
    display('Feature Elimination threshold set to 0.1');
end
%


show_network(model)

for layeri=1:(length(model.layers))
    
    
    if length(model.fe_thres)==1
        thres=model.fe_thres;
    else
        thres=model.fe_thres(layeri);
    end
    
    connections_with_high_weight=sum((abs(model.layers(layeri).W)>thres)');
    
    nodeinds=find(connections_with_high_weight>0);
    %display([num2str(model.layersizes(layeri)-length(nodeinds)) ' nodes removed from layer ' num2str(layeri)])
    model.layers(layeri).W=model.layers(layeri).W(nodeinds,:);
    model.layers(layeri).X=model.layers(layeri).X(:,nodeinds);
    
    
    
    model.layers(layeri).Ws(1)=length(nodeinds);
    model.layersizes(layeri)=length(nodeinds);
    model.layers(layeri).inds=model.layers(layeri).inds(nodeinds);
    model.layersizes(layeri)=model.layersizes(layeri);

    if layeri>1
        model.layers(layeri-1).W=model.layers(layeri-1).W(:,nodeinds);
        model.layers(layeri-1).out=model.layers(layeri-1).out(:,nodeinds);
        
        
        model.layers(layeri-1).Ws(2)=length(nodeinds);
        model.layers(layeri-1).B=model.layers(layeri-1).B(nodeinds);
        
        if isfield(model.layers(layeri-1),'grad') && ~isempty(model.layers(layeri-1).grad)
            model.layers(layeri-1).grad=model.layers(layeri-1).grad(:,nodeinds);
        end
        
        if isfield(model.layers(layeri-1),'doutdnet') && ~isempty(model.layers(layeri-1).doutdnet)
            model.layers(layeri-1).doutdnet=model.layers(layeri-1).doutdnet(:,nodeinds);
        end
        
        
    else
        model.x=model.x(:,nodeinds);
    end
    
end


if length(nodeinds)==0;
    
    error('All nodes removed, consider putting more lenient feature elimination threshold')
end

end