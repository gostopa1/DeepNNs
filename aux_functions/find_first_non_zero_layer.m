function first_non_zero_layer_index=find_first_non_zero_layer(model)

first_non_zero_layer_index=1;
%for layeri=length(model.layers):-1:1
for layeri=1:length(model.layers)
    if model.layers(layeri).lr==0
        first_non_zero_layer_index=layeri;
    end
end


end