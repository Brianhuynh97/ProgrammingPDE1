import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from  scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def grid(N, ansatz='linear'):
    # N: the number of elements
    # ansatz: linear(default) will generate 2 nodes per element
    # return: a vector containing the node coordinates
    
    
    def f(x):               #for non-uniform grid only
            return x**2
        
        
    if (ansatz=='linear'):
        #TODO
        #uniform grid
        node_count = N+1
        vec = np.linspace(0,1,node_count)
        
        #non-uniform grid
        vec = f(np.linspace(0, 1, node_count))
        
    else:
        #TODO
        #uniform grid
        node_count = 3*N-(N-1)
        vec = np.linspace(0,1,node_count)

        ##non-uniform grid
        vec = f(np.linspace(0, 1, node_count))



    return vec
#print(grid(5,ansatz='linear'))

def Phi(x, i, lattice):
    """
    Returns the value of the i-th 1D linear FEM basis function at point x.
    
    Args:
        x       : float, the point where Phi_i is evaluated
        i       : int, index of the basis function
        lattice : array-like, node positions [x0, x1, ..., xn]
        
    Returns:
        float : value of Phi_i at x
    """
    if i == 0:
        if lattice[i] <= x <= lattice[i+1]:
            h = lattice[i+1] - lattice[i]
            return (lattice[i+1] - x)/h
        else:
            return 0.0
    elif i == len(lattice) - 1:
        if lattice[i-1] <= x <= lattice[i]:
            h = lattice[i] - lattice[i-1]
            return (x - lattice[i-1])/h
        else:
            return 0.0
    else:
        h_left = lattice[i] - lattice[i-1]
        h_right = lattice[i+1] - lattice[i]
        if lattice[i-1] <= x <= lattice[i]:
            return (x - lattice[i-1])/h_left
        elif lattice[i] <= x <= lattice[i+1]:
            return (lattice[i+1] - x)/h_right
        else:
            return 0.0

def grad_Phi(x, i, lattice):
    """
    Returns the derivative of the i-th 1D linear FEM basis function at point x.
    
    Args:
        x       : float, the point where derivative is evaluated
        i       : int, index of the basis function
        lattice : array-like, node positions [x0, x1, ..., xn]
        
    Returns:
        float : derivative of Phi_i at x
    """
    # Step size to the left and right
    if i == 0:
        h_right = lattice[i+1] - lattice[i]
        if lattice[i] <= x <= lattice[i+1]:
            return 1/h_right
        else:
            return 0.0
    elif i == len(lattice)-1:
        h_left = lattice[i] - lattice[i-1]
        if lattice[i-1] <= x <= lattice[i]:
            return -1/h_left
        else:
            return 0.0
    else:
        h_left = lattice[i] - lattice[i-1]
        h_right = lattice[i+1] - lattice[i]
        if lattice[i-1] <= x <= lattice[i]:
            return 1/h_left
        elif lattice[i] <= x <= lattice[i+1]:
            return -1/h_right
        else:
            return 0.0

def assembleMatrix(lattice, ansatz='linear'):
    # lattice: node vector
    # return: stiffness matrix
    node_count = len(lattice)     #number of nodes

        
    #Matrix is NxN, 
    A = np.zeros([node_count,node_count])
    
    if (ansatz=='linear'):
        #TODO
        for i in range(0, node_count): #range(0 to node_count) because last element is not included
            for j in range(i-1, i+2): #,...i+2 because last element is not included
                if j==-1 or j==node_count: 
                    continue
                elif i==0:
                    A[i,j], err = quad(lambda x, i=i, j=j: grad_Phi(x,i,lattice)*grad_Phi(x,j,lattice), lattice[0], lattice[1])
                elif i==node_count-1:
                    A[i,j], err = quad(lambda x,i,j : grad_Phi(x,i,lattice)*grad_Phi(x,j,lattice), lattice[i-1],lattice[i], args=(i,j,))
                else:
                    A[i,j], err = quad(lambda x,i,j : grad_Phi(x,i,lattice)*grad_Phi(x,j,lattice), lattice[i-1],lattice[i+1], args=(i,j,))

    else:
        #if node_count%2==0:
            #print("only even values for N allowed")
        #TODO
        for i in range(1, node_count,2): #range(0 to node_count) because last element is not included
            for j in range(i-1, i+2): #,...i+2 because last element is not included
                #print(i,j)
                if j==-1 or j==node_count: 
                    continue
                elif i==0:
                    A[i,j], err = quad(lambda x,i,j : grad_Phi_quad(x,i,lattice)*grad_Phi_quad(x,j,lattice), lattice[i],lattice[i+1], args=(i,j,))
                elif i==node_count-1:
                    A[i,j], err = quad(lambda x,i,j : grad_Phi_quad(x,i,lattice)*grad_Phi_quad(x,j,lattice), lattice[i-1],lattice[i], args=(i,j,))
                else:
                    A[i,j], err = quad(lambda x,i,j : grad_Phi_quad(x,i,lattice)*grad_Phi_quad(x,j,lattice), lattice[i-1],lattice[i+1], args=(i,j,))
        
        for i in range(0, node_count,2): #range(0 to node_count) because last element is not included
            for j in range(i-2, i+3): #,...i+3 because last element is not included
                #print(i,j)
                if j<0 or j>node_count-1: 
                    continue
                elif i==0:
                    A[i,j], err = quad(lambda x,i,j : grad_Phi_quad(x,i,lattice)*grad_Phi_quad(x,j,lattice), lattice[i],lattice[i+2], args=(i,j,))
                elif i==node_count-1:
                    A[i,j], err = quad(lambda x,i,j : grad_Phi_quad(x,i,lattice)*grad_Phi_quad(x,j,lattice), lattice[i-2],lattice[i], args=(i,j,))
                else:
                    A[i,j], err = quad(lambda x,i,j : grad_Phi_quad(x,i,lattice)*grad_Phi_quad(x,j,lattice), lattice[i-2],lattice[i+2], args=(i,j,))
        

    return A

#node_vec = grid(3,ansatz='quad')
#print(node_vec)
#print(assembleMatrix(node_vec,ansatz="quad"))

def rhsConstant(lattice, ansatz='linear'):
    # lattice: node values
    # return: vector of right hand side values
    
    f = 1
    node_count = len(lattice)
        
    b = np.zeros(node_count)
    #print(lattice)
    
    if (ansatz=='linear'):
        #TODO
        for i in range(0, node_count): #range(0 to node_count) because last element is not included
            
            #print("i = " + str(i))
                
            #print("int")
            if i==0:
                b[i], err = quad(lambda x,i : f*Phi(x,i,lattice), lattice[i],lattice[i+1], args=(i,))
            elif i==node_count-1:
                b[i], err = quad(lambda x,i : f*Phi(x,i,lattice), lattice[i-1],lattice[i], args=(i,))
            else:
                b[i], err = quad(lambda x,i : f*Phi(x,i,lattice), lattice[i-1],lattice[i+1], args=(i,))

                
    else:
        #TODO
        for i in range(1, node_count,2): #range(0 to node_count) because last element is not included
                b[i], err = quad(lambda x,i : f*Phi_quad(x,i,lattice), lattice[i-1],lattice[i+1], args=(i,))
        for i in range(0, node_count,2): #range(0 to node_count) because last element is not included
            if i==0:
                b[i], err = quad(lambda x,i : f*Phi_quad(x,i,lattice), lattice[i],lattice[i+2], args=(i,))
            elif i==node_count-1:
                b[i], err = quad(lambda x,i : f*Phi_quad(x,i,lattice), lattice[i-2],lattice[i], args=(i,))
            else:
                b[i], err = quad(lambda x,i : f*Phi_quad(x,i,lattice), lattice[i-2],lattice[i+2], args=(i,))
    return b
#node_vec = (grid(3,ansatz='quad'))
#print(node_vec)
#print(rhsConstant(node_vec,ansatz='quad'))

def interpolate(x, lattice, u, ansatz='linear'):
    # x: a real number on the domain, where the function shall be evaluated
    # lattice: node vector
    # u: solution vector
    # return: the appoximated solution u(x) 
    #print(lattice)
    #print(u)
    
    if (ansatz=='linear'):
        assert(x<=1)
        assert(x >= 0)
        #TODO
        u_x = 0
        for i in range(1, len(lattice)-1):
            #print(i)
            #print("phi("+str(i)+") = " + str(Phi(x,i+1,lattice)))
            u_x += u[i]*Phi(x,i,lattice)
            #print(u_x)
        
        
        
    else:
        assert(x<=1)
        assert(x >= 0)
        #TODO
        u_x = 0
        for i in range(1, len(lattice)-1):
            #print(i)
            #print("phi("+str(i)+") = " + str(Phi(x,i+1,lattice)))
            u_x += u[i]*Phi_quad(x,i,lattice)
            #print(u_x)
   
    return u_x

def solConstant(x):
    # x: a real number (or a vector of real numbers), where the analytic solution is computed
    # return: a real number (or a vector of real numbers) of the analytic solution for f=1
    
    if 'float' in type(x).__name__:
        #print("float confirmed")
        X =  -0.5*(x-0.5)**2+1/8
    else:
        #print("not confirmed")
        #print(type(x).__name__) 
        X = np.zeros(len(x))
        #for i in range(0, len(X)):
        X =  -0.5*(x-0.5)**2+1/8

    return X

#a = FEM1DConstant(2, ansatz='quad')
#X = np.linspace(0,1,1000)
#Y = np.zeros(1000)
#for i in range(0,len(X)-1):
#    Y[i] = interpolate(X[i],a[0],a[1], ansatz='quad')
#plt.plot(X,Y)
def rhsDelta(lattice, ansatz='linear'):
    # lattice: node values
    # return: vector of right hand side values
    
    node_count = len(lattice)
    b = np.zeros(node_count)
        
    
    if (ansatz=='linear'):
        #TODO
        for i in range(0, node_count): #range(0 to node_count) because last element is not included
            b[i] = 2*Phi(1/2,i,lattice)
            
    else:
        #TODO
        for i in range(0, node_count): #range(0 to node_count) because last element is not included
            b[i] = 2*Phi_quad(1/2,i,lattice)
        
    return b
#print(rhsDelta(grid(5,ansatz='quad'),ansatz='quad'))


def FEM1DConstant(N, ansatz='linear'):
    # N: number of elements
    # ansatz: choose between 'linear' or 'quadratic' ansatz functions
    # return: pair (node vector, solution vector)
    
   
    # Set up the node vector
    #TODO
    
    node_vector = grid(N, ansatz)
    #print(node_vector)
    
    # Assemble stiffness matrix and right hand side
    #TODO
    
    stiffness_matrix = assembleMatrix(node_vector, ansatz)
    right_hand_side = rhsConstant(node_vector, ansatz) 
    #print(stiffness_matrix)
    #print(right_hand_side)
    
    
    # solve
    #TODO
    u = np.zeros(len(node_vector))
    G = node_vector

    u_interior = spsolve(stiffness_matrix[1:len(node_vector)-1,1:len(node_vector)-1], right_hand_side[1:len(node_vector)-1])
    #print(u_interior)
    u[1:len(G)-1] = u_interior
    
    #analytical solution:
    u_analytical = -1/2 * (G-1/2)**2 + 1/8
    
    stiffness_matrix = assembleMatrix(node_vector, ansatz)

    # Solve the interior system
    u_interior = spsolve(
        stiffness_matrix[1:-1, 1:-1],
        right_hand_side[1:-1]
    )
    return G, u, u_analytical

#n = 11
#start_time = time.time()
#a = FEM1DConstant(n) 
#print(a[0])
#print(a[1])
#print(a[2])

#print(FEM1DConstant(5,ansatz='quad')[1])

#end_time = time.time()
#time_elapsed = (end_time - start_time)
#print(time_elapsed)

def FEM1DDelta(N, ansatz='linear'):
    # N: number of elements
    # ansatz: choose between 'linear' or 'quadratic' ansatz functions
    # return: pair (node vector, solution vector)
    
    # Set up the node vector
    #TODO
    
    node_vector = grid(N, ansatz)
    #print(node_vector)
    
    # Assemble stiffness matrix and right hand side
    #TODO
    
    stiffness_matrix = assembleMatrix(node_vector, ansatz)
    right_hand_side = rhsDelta(node_vector, ansatz) 
    #print(stiffness_matrix)
    #print(right_hand_side)
    
    
    # solve
    #TODO

    u = np.zeros(len(node_vector))
    u_interior = spsolve(stiffness_matrix[1:len(node_vector)-1,1:len(node_vector)-1], right_hand_side[1:len(node_vector)-1])
    G = node_vector
    
    u[1:len(G)-1] = u_interior
    
    
    #analytical solution:
    u_analytical = np.zeros(len(G))
    for i in range(0, len(G)):
        if G[i] <= 1/2:
            u_analytical[i] = G[i]
        if G[i] > 1/2:
            u_analytical[i] = 1-G[i]
    u_analytical = u_analytical
        
    
    return G, u, u_analytical

#n = 11
#a = FEM1DDelta(n) 
#print(a[0])
#print(a[1])
#print(a[2])


#print(FEM1DConstant(5,ansatz='quad')[1])

#end_time = time.time()
#time_elapsed = (end_time - start_time)
#print(time_elapsed)

def plotResult(x, u, X, U, title='Title', ansatz='linear'):
    """
    Plot FEM solution vs analytical solution.

    Args:
        x      : node vector
        u      : FEM solution at nodes
        X      : positions to evaluate analytical solution
        U      : analytical solution values at X
        title  : plot title
        ansatz : 'linear' or 'quadratic'
    """
    # Interpolate FEM solution on the fine grid X
    u_int = np.zeros_like(X)
    for idx, val in enumerate(X):
        u_int[idx] = interpolate(val, x, u, ansatz)
    
    # Interpolated FEM solution at nodes (optional, for markers)
    u_nod = np.zeros_like(x)
    for idx, val in enumerate(x):
        u_nod[idx] = interpolate(val, x, u, ansatz)
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_axes([0,0,1,1])
    
    ax.plot(X, u_int, '-.r', label='FEM interpolated')
    ax.plot(x, u_nod, 'xr', label='FEM nodes')
    ax.plot(X, U, '-b', label='Analytical solution')
    
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.show()


x, u, u_analytical = FEM1DConstant(11)
X = np.linspace(0, 1, 1000)
U = solConstant(X)
plotResult(x, u, X, U, title='FEM 1D f = 1 (linear)')