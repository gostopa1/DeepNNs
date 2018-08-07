function [model,out]=forwardpassing_nolr(model,x)
model.layers(1).X=x;
layerno=length(model.layers);
for layeri=model.fnzl:layerno
    if sum(model.layers(layeri).lr(:))~=0
        if strcmp(model.layers(layeri).activation,'softmaxact')
            [model.layers(layeri).out, model.layers(layeri).doutdnet]=softmaxact(model.layers(layeri).X,model.layers(layeri).W,model.layers(layeri).B);
        elseif strcmp(model.layers(layeri).activation,'tanhact')
            [model.layers(layeri).out, model.layers(layeri).doutdnet]=tanhact(model.layers(layeri).X,model.layers(layeri).W,model.layers(layeri).B);
        elseif strcmp(model.layers(layeri).activation,'relu')
            [model.layers(layeri).out, model.layers(layeri).doutdnet]=relu(model.layers(layeri).X,model.layers(layeri).W,model.layers(layeri).B);
        elseif strcmp(model.layers(layeri).activation,'softsign')
            [model.layers(layeri).out, model.layers(layeri).doutdnet]=softsign(model.layers(layeri).X,model.layers(layeri).W,model.layers(layeri).B);
        elseif strcmp(model.layers(layeri).activation,'logsi')
            [model.layers(layeri).out, model.layers(layeri).doutdnet]=logsi(model.layers(layeri).X,model.layers(layeri).W,model.layers(layeri).B);
        else
            [model.layers(layeri).out, model.layers(layeri).doutdnet]=feval(model.layers(layeri).activation,model.layers(layeri).X,model.layers(layeri).W,model.layers(layeri).B);
        end
        
        %% If we are in the last layer, set the output of the model as the layer's output. Else set its output as the input to the next one.
        if layeri==layerno
            out=model.layers(layeri).out;
        else
            model.layers(layeri+1).X=model.layers(layeri).out;
            
        end
    end
end

end