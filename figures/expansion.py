import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
	print('Max Integer: {}'.format(sys.maxsize))
	# comb. expansion of trees
	nodes = 90
	nums = []
	num_funcs = 9
	num_feats = 500
	for i in np.arange(0,nodes):
		num_trees = (num_funcs**i)*catnumber(i)#*(num_feats**(2*i))
		if i > 19:
			# print(i,num_trees)
			continue
		else:
			nums.append(num_trees)

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(np.arange(0,len(nums)),nums)
	ax.axhline(y=125000000000)
	ax.set_yscale('log')
	plt.show()

def catnumber(n):
  ans = 1.0
  for k in range(2,n+1):
     ans = ans *(n+k)/k
  return ans

if __name__ == '__main__':
	main()
