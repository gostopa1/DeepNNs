function [model,out]=forwardpassing(model,x)
model.layers(1).X=x;
layerno=length(model.layers);
for layeri=1:layerno

    
    if strcmp(model.layers(layeri).activation,'softmaxact')
        [model.layers(layeri).out, model.layers(layeri).doutdnet]=feval(model.layers(layeri).activation,model.layers(layeri).X,model.layers(layeri).W,model.layers(layeri).B,model.target);
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