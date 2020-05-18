import sys
from scoop import futures

def square(x):
	return x*x

if __name__ == "__main__":
	# obtain n from first command line argument
	n = int(sys.argv[1])
	k = 1000
	# compute the square of the first n numbers, in parallel using SCOOP functionality
	squares = futures.map(square, range(k)) # note: returns an iterator
	print("Trial %d - First %d squares: %s" % (n, k, list(squares)))