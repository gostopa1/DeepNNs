function [model,out]=forwardpassing(model,x)
model.layers(1).X=x;
for layeri=1:(length(model.layers)-1)
    
    model.layers(layeri+1).X=relu(model.layers(layeri).X,model.layers(layeri).W,model.layers(layeri).B); % Assign as input to the next layer, the output of the current layer
    
    if strcmp(model.layers(layeri).activation,'softmaxact')
        [model.layers(layeri).out, model.layers(layeri).doutdnet]=feval(model.layers(layeri).activation,model.layers(layeri).X,model.layers(layeri).W,model.layers(layeri).B,model.target);
    else
        [model.layers(layeri).out, model.layers(layeri).doutdnet]=feval(model.layers(layeri).activation,model.layers(layeri).X,model.layers(layeri).W,model.layers(layeri).B);
    end
end

layeri=length(model.layers);

if strcmp(model.layers(layeri).activation,'softmaxact')
    [model.layers(layeri).out, model.layers(layeri).doutdnet]=feval(model.layers(layeri).activation,model.layers(layeri).X,model.layers(layeri).W,model.layers(layeri).B,model.target);
else
    [model.layers(layeri).out, model.layers(layeri).doutdnet]=feval(model.layers(layeri).activation,model.layers(layeri).X,model.layers(layeri).W,model.layers(layeri).B);
end

out=model.layers(layeri).out;

end