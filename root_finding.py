try:
    import numpy as np
except:
    raise RuntimeError("This Library Reuires Numpy")

def bisection_solve(
    function,
    x0,
    x1,
    rtol=1e-6
):
    '''
    This function implements the Bisection Routine on  a provided function *function* that it takes in as an argument.
    
    Arguments:
    ========= 
    
    function: Any Real valued function defined on the interval [x0, x1].

    x0: A number in the domain of the provided function.

    x1: A number in the domain of the provided function.

    ** NOTE: The function has to be positive at exactly one of x0 or x1 and negative at exactly one of x0 and x1**
    ** NOTE: The solver is biased towards x0, so it prefers converging towards x0 as compared to x1 **

    rtol: This adjusts the relative tolerance of the solution routine. The solution loop continues until the
          number eps = |x1 - x0| < rtol.
    
    '''
    [z1, z2]  = np.vectorize(function)(np.array([x0, x1]));
    if z1 * z2 >= 0:
        raise RuntimeError("Please provide a pair of x Values Such that " +
                           "the provided function has a positive value at end " + 
                           "has a negative value at the other end. Also ensure " +
                           "that neither of the endpoints is a zero of the provided function")
        
    # Setup the Solver Loop
    x0_current = x0;
    x1_current = x1;
    eps = abs(x0_current - x1_current);
    iterations = 0;
    
    while eps > rtol:
        iterations += 1;
        eps = abs(x1_current - x0_current)
        
        # Bisection Algorithm
        phi_0 = function(x0_current);
        phi_1 = function(x1_current);
        phi_2 = function((x0_current + x1_current)/2)

        if (phi_0 * phi_2) <= 0:
            x1_current = 1/2 * (x0_current + x1_current);
        else:
            x0_current = 1/2 * (x0_current + x1_current);
    
    eps = abs(x1_current - x0_current)    
    return [x0_current, x1_current, eps, iterations]

def incremental_search(
        function,
        xmin ,
        xmax ,
        step = 1e-2
    ):

    """
    Implementation of the incremental search routine.

    Arguments:
    ==========


    function:
    --------
        A function defined on the interval xmin, xmax.

    xmin:
    -----
        The lower bound on the interval on which you want to find the zeros.

    xmax:
    -----
        The upper bound on the interval on which you want to find the zeros.

    step:
    ----
        step is the difference between successive partitions of the interval [xmin, xmax] carried out by 
        the function.
    """

    if (xmin > xmax):
        raise RuntimeError("xmin should not be greater than xmax.")

    X_Range = np.arange(xmin, xmax, step);
    Fun_Range = np.vectorize(function)(X_Range)

    Zeros = Fun_Range[1:]* Fun_Range[:-1]
    Zeros = X_Range[:-1][Zeros <= 0]

    return Zeros

def incremental_bisection(
    function,
    xmin,
    xmax,
    step,
    rtol = 1e-6
):

    """
    Incremental Bisection Solver With Specified Tolerance.

    Description:
    -----------

    This solver employs the incremental search algorithm to calculate the location of zeros and then uses 
    the bisection method to fix each of the values within a specifed error tolerance.

    Note:
    -----

    If the algorithm fails, first try to decrease the value of step.

    Arguments:
    =========

    function:
    --------
        A function defined on the interval xmin, xmax.

    xmin:
    -----
        The lower bound on the interval on which you want to find the zeros.

    xmax:
    -----
        The upper bound on the interval on which you want to find the zeros.

    step:
    ----
        step is the difference between successive partitions of the interval [xmin, xmax] carried out by 
        the function.
    

    rtol: 
    ----
          This adjusts the relative tolerance of the solution routine. The solution loop continues until the
          number eps = |x1 - x0| < rtol.
    
    
    """
   
    _Zeros = incremental_search(function, xmin, xmax, step);

    # Bottleneck here
    Zeros_ISearch = []
    for i in _Zeros:
        Zeros_ISearch.append(
            (
                i + step,
                i - step
            )
        )
    del _Zeros 

    # Iterate Over Each Zero
    Zeros = []
    for i in Zeros_ISearch:
        Zeros.append(
            bisection_solve(
                function,
                i[0],
        i[1],
                rtol
            )
        )
    
    del Zeros_ISearch

    # Finish 
    return Zeros

def fixed_point_iteration(
        function,
        x0,
        steps=int(1e3),
        rtol=1e-6
    ):

   """
   Fixed Point Iteration Scheme.

   Arguments:
   =========

   function:
   --------
       A real function, preferably without singularities.

   x0:
   --
       Initial Guess to the location of the root.

   steps:
   -----
       Number of fixed point iterations.
       Pass this as np.inf and the solver will try to achieve rtol.

   rtol:
   -----
       rtol is the tolearnce of the solver. This option is only relevant if steps =np.inf.
   """


   xprev, xcurr = x0, x0
   for i in range(steps):
       if steps==np.inf and abs(xprev-xcurr) < rtol:
               break

       xcurr = xprev
       xcurr = function(xprev)

   return xcurr

def newton_raphson(function, x0, rtol=1e-6):
    """
    This function proviedes an implementation of the Newton Raphson Technique.

    Arguments:
    ---------
    
    function:
    --------
        A real differentiable function. 
    x0:
    --
       An Initial Guess to the solution value. Preferably, the function should never have a maximum or minimum close to x0.

    rtol: 
    ---
        Relative Tolerance Between Two Successive values generated 
        by the NR routine. Note that this is not reliable for functions
        with a small derivative.

    The derivative is calculated using a 2nd order accurate uniform finite difference scheme.
    
    """


    # Second Order Accurate Central Derivative
    dfdx = lambda x: 1/(2 * h) * (function(x+h) - function(x - h))

    h = 1e-5
    x_curr = x0 + rtol
    x_new = x0

    while(np.abs(x_new - x_curr) > rtol):
        x_curr = x_new
        x_new -= function(x_new)/dfdx(x_new)

    return x_new

def secant_solve(function, x0, h = 1e-2, rtol = 1e-6):
    """

    Uses the Modified Secant Method to Solve the algebraic equation f(x) = 0. Using an Initial Guess x0.

    Arguments:
    ---------

    function:
    --------
        A real differentiable function. 
    
    x0:
    --
       An Initial Guess to the solution value. Preferably, the function should never have a maximum or minimum close to x0.

    h:
    --
        h is the step size for the secant method. A smaller value of h than the default might be a good idea.

    rtol: 
    ---
        Relative Tolerance Between Two Successive values generated 
        by the NR routine. 

    """

    x_new = x0
    x_old = x0+rtol
    while(np.abs(x_new - x_old) > rtol):
        x_old = x_new
        x_new -= h * function(x_new) / (function(x_new + h) - function(x_new));

    return x_new

def IQI_solve(function, x0, x1, x2, rtol=1e-6):

    X = [x0, x1, x2]
    Y = [function(X[i]) for i in range(3)]

    return root_new

def brent_solve():
    pass
