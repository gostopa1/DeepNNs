clear
addpath('./activation_functions')
addpath('./loss_functions')

x = [1 1; 1 0; 0 1; 0 0];
y(:,2) = [1; 0; 0; 1];
y(:,1) = [0; 1; 1; 0];
N=size(x,1);

N=1000; x=rand(N,2); y=xor(x(:,1)<0.5,x(:,2)<0.5); y(:,2) = 1 - y(:,1);

x=zscore(x,[],1);
%y=zscore(y,[],1)

[x_test,y_test]=meshgrid(0:0.1:1,0:0.1:1);

%test_data=[x_test(: ) y_test(:)];
test_data=zscore([x_test(: ) y_test(:)],1);

%% The next part is identical in regression and classification
clear model

layers=[3];
batchsize=100
noins=size(x,2);
noouts=size(y,2);

layers=[noins layers noouts];
lr=0.1; activation='tanhact';
%lr=0.1; activation='relu';
lr=0.1; activation='logsi';

model.layersizes=[layers];
model.target=y;
for layeri=1:(length(layers)-1)
    model.layers(layeri).lr=lr;
    model.layers(layeri).blr=lr;
    model.layers(layeri).Ws=[layers(layeri) layers(layeri+1)]
    model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1))-0.5)/10;
    model.layers(layeri).B=(randn(layers(layeri+1),1)-0.5)/10;
            
    model.layers(layeri).activation=activation;
    
end
layeri
%model.layers(layeri).lr=lr/10; model.layers(layeri).activation='linact';
%model.layers(layeri).lr=lr/10; model.layers(layeri).activation='tanhact';


clear error
epochs=1000
for epoch=1:epochs
    display(['Epoch: ' num2str(epoch)])
    %%Forward passing
    
    [model,out(:,:,epoch)]=forwardpassing(model,x);
    a(epoch)=(model.layers(end).out(1,1));
    preact=model.layers(end).X*model.layers(end).W+repmat(model.layers(end).B,1,size(model.layers(end).X,1))';
    a(epoch)=preact(1,1);
    [error(epoch),dedout]=quadratic_cost(model); % Using quadratic error function
    
    [error(epoch),dedout]=cross_entropy_cost(model); % Using quadratic error function
    randomized_sample_indices=randperm(N);
    for batchi=1:batchsize:N
        layeri=length(model.layers);
               
        batchinds=randomized_sample_indices(batchi:(batchsize+batchi-1));
        
        clear dedw dedb
        
        ins=model.layers(layeri).Ws(1);
        outs=model.layers(layeri).Ws(2);
                
        dnetdw=permute(repmat(model.layers(layeri).X(batchinds,:),1,1,outs),[2 3 1]);
        dedb=(dedout(batchinds,:).*model.layers(layeri).doutdnet(batchinds,:))';
        dedw=permute(repmat(dedb,1,1,ins),[3 1 2]).*dnetdw;
        
        %model.layers(layeri).grad=dedout.*model.layers(layeri).doutdnet;
        model.layers(layeri).grad=dedb';
        
        %model.layers(layeri).W=model.layers(layeri).W-model.layers(layeri).lr*sum(dedw,3);
        %model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr*sum(dedb,2);
        
        
        model.layers(layeri).W=model.layers(layeri).W-model.layers(layeri).lr*mean(dedw,3);
        model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr*mean(dedb,2);
        
        for layeri=(length(model.layers)-1):-1:1
            clear dedw dedb
            ins=model.layers(layeri).Ws(1);
            outs=model.layers(layeri).Ws(2);
            
            %model.layers(layeri).grad=squeeze(sum(permute(repmat(model.layers(layeri+1).grad,1,1,outs),[3 2 1]).*repmat(model.layers(layeri+1).W,1,1,batchsize),2))'.*model.layers(layeri).doutdnet(batchinds,:);
            model.layers(layeri).grad=permute(sum(permute(repmat(model.layers(layeri+1).grad,1,1,outs),[3 2 1]).*repmat(model.layers(layeri+1).W,1,1,batchsize),2),[1 3 2])'.*model.layers(layeri).doutdnet(batchinds,:);
            
            dedb=model.layers(layeri).grad';
            dedw=permute(repmat(dedb,1,1,ins),[3 1 2]).*permute(repmat(model.layers(layeri).X(batchinds,:),1,1,outs),[2 3 1]);
                        
            %model.layers(layeri).W=model.layers(layeri).W-model.layers(layeri).lr*sum(dedw,3);
            %model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr*sum(dedb,2);
            
            model.layers(layeri).W=model.layers(layeri).W-model.layers(layeri).lr*mean(dedw,3);
            model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr*mean(dedb,2);        
        end
    end
    
    if mod(epoch,20)==1
        %model.layers(end).W
        %model.layers(end).grad
        %pause
        
        show_network
        drawnow
    end
end

%print('a.png','-dpng','-r500')
%%

model.test=0;
[model,out_test]=forwardpassing(model,[test_data]);
factor=15;
figure(1)
%clf
subplot(4,1,3)
hold on

for pointi=1:size(test_data,1)
    mat(((x_test(pointi)+1)*10)-9,((y_test(pointi)+1)*10)-9)=out_test(pointi,1);
end
contour(mat,[0.25 0.5 0.75],'LineWidth',3)
for pointi=1:size(test_data,1)
    if out_test(pointi,1)>0
        dotcol='k';
    else
        dotcol='r';
    end
    m=plot(((x_test(pointi)+1)*10)-9,((y_test(pointi)+1)*10)-9,[dotcol '.']);
    set(m,'MarkerSize',abs(out_test(pointi,1))*factor+5)
end
set(gca,'XTick',[1 11],'XTickLabel',[0 1]);
set(gca,'YTick',[1 11],'YTickLabel',[0 1]);

%axis off
%box off
subplot(4,1,4)
plot(error)
xlabel('Epoch')

ylabel('Error')



%%

[~,indpre]=max(out(:,:,end)');

%indpre-1
[~,indtarget]=max(model.target');
perf=(sum(indpre==indtarget)/length(indpre))*100;
display([sprintf('Performance : %3.2f%%',perf)])
