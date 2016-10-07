function display(x,y,o,trainOrTest,zoom,freqSecs)
	if imgDisplay == nil then 
		local initPic = torch.rand(100,200):reshape(100,200)
		imgDisplay0 = image.display{image=initPic, zoom=zoom, offscreen=false}
		imgDisplay1 = image.display{image=initPic, zoom=zoom, offscreen=false}
		imgDisplay = 1 
		displayTimer = torch.Timer()
	end

	if displayTimer:time().real > freqSecs  then

		local choose = torch.random(x:size(1))
		local x1 = x:narrow(1,choose,1):squeeze() -- input
		local o1 = o:narrow(1,choose,1):double() -- prediction
		local y1 = y:narrow(1,choose,1):double() -- truth

		local o1Up = image.scale(o1:squeeze():double(),x1:size(3),x1:size(2)):resize(1,x1:size(2),x1:size(3)) -- upscale prediciton
		local o1Up = o1Up:repeatTensor(3,1,1) -- repeat for rgb
		local av = torch.add(o1Up,x1:double())/2 -- average prediction and input
		--local inter = torch.cmul(o1Up,x1:double())/2
		local interAv = torch.cat(av,o1Up)

		local title

		if trainOrTest == "train" then
			title = "Train"
			image.display{image = torch.cat(y1,o1), win = imgDisplay0, legend = title}
		else 
			title = y[choose] -- y is names if  we dont have y
		end
		image.display{image = interAv, win = imgDisplay1, legend = title}

		collectgarbage()
		displayTimer:reset()
	end
end
