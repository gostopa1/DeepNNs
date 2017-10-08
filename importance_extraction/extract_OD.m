function impos=extract_OD(model,x)
[~,out]=forwardpassing(model,x);

for i=1:size(x,2)
x2=x;
x2(:,i)=0;
[~,outz]=forwardpassing(model,x2);
impos(:,i,:)=out-outz;

end