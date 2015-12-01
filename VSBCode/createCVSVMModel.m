function SVMModel = createCVSVMModel(dataX, dataY)
c = cvpartition(length(dataY),'KFold',10);

minfn = @(z)kfoldLoss(fitcsvm(dataX,dataY,'CVPartition',c,...
    'KernelFunction','rbf','BoxConstraint',exp(z(2)),...
    'KernelScale',exp(z(1))));

opts = optimset('TolX',5e-4,'TolFun',5e-4);
m = 20;
fval = zeros(m,1);
z = zeros(m,2);
for j = 1:m;
    'Processing...'
    [searchmin, fval(j)] = fminsearch(minfn,randn(2,1),opts);
    z(j,:) = exp(searchmin);
end

z = z(fval == min(fval),:);

SVMModel = fitcsvm(dataX,dataY,'KernelFunction','rbf',...
    'KernelScale',z(1),'BoxConstraint',z(2));

end