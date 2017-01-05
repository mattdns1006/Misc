require "nn"

nFeats = 3
xTr = torch.rand(10,nFeats)
x = torch.rand(1,nFeats)
bn = nn.BatchNormalization(nFeats)
bn.weight:fill(1)
bn:forward(xTr)
eps = bn.eps

mu = bn.running_mean
var = bn.running_var

bn:evaluate()
truth = bn:forward(x)
print(truth)
for i=1, 3 do
	feat = x[1][i]
	mu1 = mu[i]
	std1 = var[i]
	out = (feat - mu1)/torch.sqrt(std1+eps)
	print(i,feat,out)
end
