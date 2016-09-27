function joinTables(t1,t2)
	local t3 = {}
	for i=1, #t1 do
		t3[#t3+1] = t1[i]
	end
	for i=1, #t2 do
		t3[#t3+1] = t2[i]
	end
	return t3
end
