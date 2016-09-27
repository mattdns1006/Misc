function shuffle(list)
	local indices = {}
	for i = 1, #list do
		indices[#indices+1] = i
	end

	-- create shuffled list
	local shuffled = {}
	for i = 1, #list do
		local index = math.random(#indices)

		local value = list[indices[index]]

		table.remove(indices, index)

		shuffled[#shuffled+1] = value
	end

	return shuffled
end
                                                                                                             
