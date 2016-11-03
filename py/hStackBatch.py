import numpy as np

def hStackBatch(batch,hStack=1):
	'''
	For displaying a batch of images nicely
	'''
	if hStack == 1:
		stack = np.hstack
	else:
		stack = np.vstack
	batch = batch.squeeze()
	batchSize = batch.shape[0]
	x = batch[0]
	for i in range(1,batchSize):
		x = stack((x,batch[i]))
	return x
