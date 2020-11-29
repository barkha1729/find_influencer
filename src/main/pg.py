import numpy as np
from  scipy import sparse
import random 
import networkx as nx
import matplotlib.pyplot as plt
from itertools import count
import time

Filename = "Wiki-Vote.txt"
NODES = 9000
EDGES = 103689


#function to compute Page Rank for the given adjacency matrix of the graph
#beta and epsilon is feeded into the function as constants
#return the number of iterations and rank matrix
def Estimate_PageRank(M, beta=0.85, epsilon=10**-4):

#as given matrix is adjacency matrix
#adjusting the matrix to get the stochastic adjacency matrix

	#calculating the outdegrees of each node with the help of sum function
	#incorporating even the beta value here i.e. di/beta
	#Transpose is needed as sum at axis=0 returns row vector but ranks are column vectors 
	OutDegree_beta = M.sum(axis=0).T/beta
	n,_ = M.shape					#number of nodes
	ranks = np.ones((n,1))/n 		#Initial rank matrix to start the iterations i.e. 1/n
	iterations = 0					#number of iterations
	check = True					#Variable to loop the iteration

	#Declarations reqiured for Graph Visualisation:
	G=nx.DiGraph()					#Directed Graph using networkx for plotting the visualisations
	nodes=[]						#Stores the list of random nodes generated for plotting the graph
	size=[]							#List of sizes of the nodes which is scaled up factor of their ranks
	labels={}						#Labels of the nodes which is scaled up factor of their ranks

	#while loop for Page rank iterations
	while check:        
	    iterations +=1
	    with np.errstate(divide='ignore'):				#Ignore division by 0 on ranks/OutDegree_beta
	    	#newranks=Beta*M*r(t-1)
	    	rank_next = M.dot((ranks/OutDegree_beta))	#Multiplying the ranks from previous iterations here
	        											#Injecting Stochastic behaviour by dividing rank with earlier computed outdegrees
	    
	    #Next update ranks with (1-Beta)*r(t-1)/N

	    #rank_sum stores the matrix after the 
	    rank_sum=np.ones((n,1))*ranks.sum()
	    rank_next=rank_next+rank_sum*(1-beta)/n

	    #Stop condition: if difference is less than epsilon then stop iterating
	    if np.linalg.norm(ranks-rank_next,ord=1)<=epsilon:
	        check = False        
	    ranks = rank_next
	    
	
	for x in range(100):					 #Generating 100 random nodes
		t=random.randint(1,8000)
		nodes.append(t)						 #appending the nodes in list
		size.append(10000000000*ranks[t])	 #storing their corresponding ranks as sizes
		labels[t]=(int)(10000000000*ranks[t])#labels as scaled up ranks

	for key in nodes:						 #adding the nodes and their corresponding edges in G garph
	 	G.add_node(key)
	 	for j in nodes:
	 		if M[j,key]==1:					 #edges are given by the adjacency matrix
	 			G.add_edge(key,j)
	'''
	#to remove isolated nodes from the graph
	for x in nx.isolates(G):
		G.remove_node(x)
		nodes.remove(x)
		size.remove(10000000000*ranks[x])
		del labels[x]
	'''

	nx.draw_random(G,node_list=nodes,node_size=size,labels=labels,with_labels=True)
	plt.show()
	return(ranks,iterations)

    



def sparse_Dic():
    with open(Filename,'r') as f:
        sp_dic_mat = sparse.dok_matrix((NODES,NODES),dtype=np.bool)
        for line in f.readlines()[4:]:
            from_node, to_node = (int(x)-1 for x in line.split())
            sp_dic_mat[to_node,from_node]=True    
    return(sp_dic_mat.tocsr())


print('Estimating page rank:')
sp_csr=sparse_Dic()
start_time = time.time()
pr , iterate = Estimate_PageRank(sp_csr)
print('\nIterations: {0}'.format(iterate))
print('Node with maximum pagerank is: {0}'.format(np.argmax(pr)+1))
print("%s"% (time.time()-start_time))