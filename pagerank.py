import numpy
import gzip
import sys

def pagerank(filename, lam, tau):
	with gzip.open(filename, 'r') as f:
		# Read all links in file line by line
		L = f.readlines()

		# Dictionary of all pages and where they link to (Q)
		P = dict()
		# Dictionary of page index mappings in I and R arrays
		M = dict()
		m_index = 0
		# Dictionary of inlink counts for each page key
		C = dict()

		# For all links
		for i in range(0, len(L)):
			source, destination = L[i].split() # Separate into source and destination

			# If source has not been recorded in P
			if source not in P.keys():
				P[source] = set() 			# Record source as key with empty set as value
				P[source].add(destination) 	# Add destination to that set
				M[source] = m_index 		# Record source as key in M, with value m_index (position in I and R)
				m_index += 1 				# Increment m_index for next page recorded

			# Otherwise, if source has been recorded in P
			else:
				P[source].add(destination) 	# Add destination to its value set

			# If destination has not been recorded in P
			if destination not in P.keys():
				P[destination] = set() 		# Record destination as key with empty set as value
				M[destination] = m_index 	# Record destination as key 
				m_index += 1				# Increment m_index for next page recorded

			if destination not in C.keys():
				C[destination] = 1
			else:
				C[destination] += 1

		size = len(P) # Length of P for calculations later
		I = numpy.zeros(shape = (size)) # Array I stores all page ranks from the previous iteration
		R = numpy.zeros(shape = (size))	# Array R stores all page ranks from the current iteration

		for i in range(0, size):
			I[i] = 0		# Initialize all cells in I to 0

		for i in range(0, size):
			R[i] = 1 / size # Initialize all cells to 1/|P|

		converged = False	# Whether ranks have converged or not

		while not converged:
			for i in range(0, size):
				I[i] = R[i] 		# Copy values of R into I

			for i in range(0, size):
				R[i] = lam / size 	# Reassign all values of R back to lambda/|P|

			q0_accumulator = 0		# Accumulator variable for when set Q (destinations from P) is empty

			# For all keys in P (all pages)
			for k in P.keys():
				p_index = M[k]		# Index of current page p
				Q = P[k]			# Set of all destinations p goes to

				# If Q is not empty
				if len(Q) > 0:
					# For each destination q from source p
					for q in Q:
						q_index = M[q] # Index of current destination q
						R[q_index] = R[q_index] + (1 - lam) * I[p_index] / len(Q) # Rq ← Rq + (1 − λ)Ip/|Q|
				else:
					q0_accumulator += (1 - lam) * I[p_index] / size # Rq ← Rq + (1 − λ)Ip/|P|

			total_diff = 0 		# Total differences between all I and R values

			for i in range(0, size):
				R[i] = R[i] + q0_accumulator 	# Add accumulated value to R[i]
				total_diff += abs(I[i] - R[i])	# Record difference from corresponding I[i] value

			if total_diff < tau:
				converged = True
				print("Converged")

		R_copy = numpy.zeros(shape = (size))

		for i in range(0, size):
			R_copy[i] = R[i]

		inlinksfile = open("inlinks.txt", "w")
		pagerankfile = open("pagerank.txt", "w")
		for i in range(1, 51):
			max_index = numpy.argmax(R_copy)
			max_value = R_copy[max_index]
			for key in M.keys():
				if M[key] == max_index:
					pagerankfile.write(key.decode("utf-8"))
			pagerankfile.write(" " + str(i) + " " + str(max_value) + "\n")
			R_copy[max_index] = 0

		for i in range(1, 51):
			max_value = max(C.values())
			keys = [key for key, v in C.items() if v == max_value]

			for k in keys:
				inlinksfile.write(k.decode("utf-8") + str(i) + str(max_value) + "\n")
				C.pop(k)

		inlinksfile.close()
		pagerankfile.close()


# MAIN ################################################
lam = 0.2
tau = 0.02

if len(sys.argv) > 2:
	lam = sys.argv[1]
	tau = sys.argv[2]

pagerank('links.srt.gz', lam, tau)
######################################################################