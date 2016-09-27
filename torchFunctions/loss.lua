require "gnuplot"

local Loss = {}
Loss.__index = Loss

function Loss.new(movingAverage)
	local self = {}
	self.current = {}
	self.movingAverage = {}
	self.maNumber = movingAverage
	return setmetatable(self,Loss)
end

function Loss:add(loss)
	table.insert(self.current,loss)
	if #self.current == self.maNumber then
		table.insert(self.movingAverage,torch.Tensor(self.current):mean())
		self.current = {}
	end
end

function Loss:plot(fig)
	assert(#self.movingAverage > 0,"Not enough observations to plot moving average.")
	local losses = torch.Tensor(self.movingAverage)
	gnuplot.figure(fig)
	gnuplot.plot(losses)
end

return Loss

