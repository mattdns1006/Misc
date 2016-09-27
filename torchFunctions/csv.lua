require "paths"

csv = {}

function csv.length(csv)
  	local count = 0
      	for _ in pairs(csv) do count = count + 1 end
        return count
end

function csv.read(path,header)
	local csvFile = io.open(path,"r")
	if header == 1 then
		local h = csvFile:read()
	end
	local data = {}

	local i = 0  
	for line in csvFile:lines('*l') do  
		i = i + 1
		data[i] = line 
	end
	return data
end

return csv 
