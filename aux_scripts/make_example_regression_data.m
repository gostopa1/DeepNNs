% This make simple regression data to approximate either hyperbolic tangent
% function or quadratic function for demontration purposes
N=500;
x=randn(N,1);

N=length(x);
y=x.^2;
%y=tanh(x)
x=x/max(x(:)); %  Normalizing

