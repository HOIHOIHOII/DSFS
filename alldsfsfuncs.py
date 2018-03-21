#alldsfsfuncs.py
from __future__ import division
from collections import Counter
import math
import numpy as np
import random
from scipy.special import erf
from matplotlib import pyplot as plt


def vector_add(v,w):
    return [v_i + w_i for v_i, w_i in zip(v,w)]
        

def vector_sum(vectors):
    return reduce(vector_add, vectors)
    

def mdim(A):
    return len(A), len(A[0])

def vector_sum(A,B):
    #compute vector sum of A and B
    return [x+y for x,y in zip(A,B)]

def scalar_mult_vector(a, V):
    return [a*i for i in V]

def scalar_mult(a, M):
    return [scalar_mult_vector(a,V) for V in M]

def dot(A,B):
    #compute dot product of A and B
    
    return sum(x*y for x,y in zip(A,B))

def madd(A,B):
    #matrix add A and B
    return [vector_sum(Ai,Bi) for Ai,Bi in zip(A,B)]

def mtrans(A):
    #get transpose matrix of A
    
    n, m = mdim(A) #A is an nxm matrix
    At = [[0]*n for _ in range(m)]  #initialise mxn matrix At 
    for i,Ai in enumerate(A):
        for j, Aij in enumerate(Ai):
            At[j][i] = Aij #put element in position i,j in A into position j,i in At
    return At
    
def mId(n):
    #create nxn identity matrix
    
    I = [[0]*n for _ in range(n)]  #initialise mxn matrix At
    for i in range(n):
        I[i][i] = 1
    return I
    
def mmult(A,B):
    #matrix multiply A and B
    
    An, Am = mdim(A)
    Bn, Bm = mdim(B)
    if Am != Bn:
        print "dims don't match"
        return None
    else:
        C = [[0]*Bm for _ in range(An)] #initialise An x Bm result matrix
        Bt = mtrans(B)
        for i in range(An):
            for j in range(Bm):
                C[i][j] = dot(A[i],Bt[j])
        return C


def det(A):
    n,m = mdim(A)
    if (n,m) != (2,2):
        print "Matrix wrong size"
    else:
        return A[0][0]*A[1][1]-A[0][1]*A[1][0]


def minvert(A):
    #invert 2x2 vector A 
    a,b = A[0]
    c,d = A[1]
    B = [[d,-b],[-c,a]]
    return scalar_mult(1/det(A),B)
    
def normal(x, mu=0, sigma=1):
    
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))


def normal_cdf(z, mu=0, sigma=1):
    
    if mu != 0 or sigma != 1:
        return normal_cdf((z-mu)/sigma)
    
    return (1+erf(z/np.sqrt(2)))/2

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.0000001):
    """find approximate inverse using binary search"""
    
    #if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    
    lo_z, lo_p =-12, 0 #normal_cdf(-10) is very close to 0 
    hi_z, hi_p = 12, 1 
    #normal_cdf(10) is very close to 1
    while hi_z-lo_z > tolerance:
        mid_z = (hi_z + lo_z)/2
        mid_p = normal_cdf(mid_z)
        if p > mid_p:
            lo_z, lo_p = mid_z, mid_p
        elif p < mid_p:
            hi_z, hi_p = mid_z, mid_p
        else:
            break
    return mid_z


def normal_probability_above(lo, mu=0, sigma=1):
    return 1 - normal_cdf(lo, mu, sigma)

def normal_probability_between(lo, hi, mu=0, sigma=1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

def normal_probability_outside(lo, hi, mu=0, sigma=1):
    return 1- normal_probability_between(lo, hi, mu, sigma)

def normal_upper_bound(p, mu=0, sigma=1):
    return inverse_normal_cdf(p, mu, sigma)

def normal_lower_bound(p, mu=0, sigma=1):
    return inverse_normal_cdf(1-p, mu, sigma)

def normal_two_sided_bound(p, mu=0, sigma=1):
    tail_probability = (1 -p)/2
    
    return normal_upper_bound(tail_probability, mu, sigma), normal_lower_bound(tail_probability, mu, sigma)

def two_sided_p_value(x, mu=0, sigma=1 ):
    dev = np.abs(mu-x)
    return 1-normal_probability_between(mu-dev,mu+dev, mu, sigma)


def make_matrix(num_rows, num_cols, entry_fn):
    """returns a num_rows x num_cols matrix
    whose (i,j)th entry is entry_fn(i,j)"""
    return [[entry_fn(i,j)
               for j in range(num_cols)]
               for i in range(num_rows)]

def dot(v,w):
    """dot product of v and w"""
    return sum(v_i*w_i for v_i,w_i in zip(v,w) )

def sum_of_squares(v):
    return sum(v_i**2 for v_i in v)

def mean(v):
    return sum(v)/len(v)
    
    
def median(v):
    """finds the middle-most value of v"""
    n= len(v)
    sorted_v = sorted(v)
    midpoint = n // 2
    
    if n % 2 == 1:
        #if odd, return the middle value
        return sorted_v[midpoint]
    
    if n % 2 == 0:
        left = sorted_v[midpoint-1]
        right = sorted_v[midpoint]
        return (left-right/2)

def quantile(x, p):
    """get the pth percentile value of x"""
    n = len(x)
    position = int(n * p)
    sorted_x = sorted(x)
    return sorted_x[position]

def mode(x):
    """returns a list, might be more than one mode"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.iteritems()
           if count == max_count]

#range already means something in python, so we'll use a different name

def data_range(x):
    return max(x) - min(x)

def de_mean(x):
    """translate x by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def variance(x):
    """assumes x has at least two elts"""
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n-1) 

def standard_deviation(x):
    return math.sqrt(variance(x))

def interquartile_range(x):
    return quantile(x,0.75)-quantile(x,0.25)

def covariance(x,y):
    n=len(x)
    return dot(de_mean(x),de_mean(y)) / (n-1)

def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x/ stdev_y
    else:
        return 

def shape(A):
    num_rows = len(A)
    num_columns = len(A[0]) if A else 0
    return num_rows, num_columns

def get_row(A,i):
    return A[i]

def get_column(A,j):
    return [A_i[j] for A_i in A]

#implement other statistics functions from DSFS


def correlation_matrix(data):
    """returns the num_columnss x num_columns matrix whose (i,j)th entry is 
    the correlation coefficient between columns i and j of data"""
    _, num_columns= shape(data)
    
    def matrix_entry(i,j):
        return correlation(get_column(data,i),get_column(data, j))[0]
    
    return make_matrix(get_column(data,i),get_column(data,j), matrix_entry)

def distance(v,w):
    return math.sqrt(sum((v_i-w_i)**2 for v_i,w_i in zip(v,w)))

def magnitude(w):
    return math.sqrt(dot(w,w))

def partial_difference_quotient(f,v,i,h):
    """returns the approximation to the ith gradient component of the multivariate function f at the point v, for small h"""
    
    w = [v_j +(h if j==i else 0) for j, v_j in enumerate(v)]
    return (f(w)-f(v))/h   

def step(v, direction, step_size):
    """move step_size in direction from v"""
    return [v_i +step_size*direction_i for v_i,direction_i in zip(v,direction)]

def sum_of_squares_gradient(v):
    return [2*v_i for v_i in v] 

def safe(f):
    """return a new function that's the same as f but outputs infty whenever f produces an error"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f

def minimise_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """Use gradient descent to find theta that minimises target function"""
    step_sizes = [1,0.1,0.01,0.001,0.0001,0.00001]
    
    theta = theta_0
    target_fn = safe(target_fn)
    gradient_fn = safe(gradient_fn)
    value = target_fn(theta)
    

    
    while True:
        gradient= gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size) for step_size in step_sizes]
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)
        
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

def negate(f):
    """return a function that for any input x returns -f(x)"""
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    """the same when f returns a list of numbers"""
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def maximise_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimise_batch(negate(target_fn),
                          negate_all(gradient_fn), 
                          theta_0, 
                          tolerance
                          )


def in_random_order(data):
    """generator that returns data in some random order"""
    indexes = [i for i,_ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]

def scalar_multiply(a,v):
    return [a*v_i for v_i in v]
        
def vector_subtract(v,w):
    return [v_i - w_i for v_i,w_i in zip(v,w)]
        
def minimise_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    """minimises target function one parameter at a time?"""
    data = zip(x,y)
    theta = theta_0                           #initial guess of input that yeilds minimum value
    alpha = alpha_0                           #initial step size
    min_theta, min_value = None, float("inf") #current minimum
    iterations_with_no_improvement = 0
    
    
    #if we ever go 100 iterations with no improvement, stop
    
    
    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)
        
        if value < min_value:
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            #otherwise not improving, so try shrikning step size
            iterations_with_no_improvement += 1
            alpha *= 0.9
            
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
            
    return min_theta

def accuracy(tp,fp,fn,tn):
    """fraction of results that are correct predictions"""
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct/total



def precision(tp,fp,fn,tn):
    """Fraction of prediction "true" that are correct"""
    return tp/(tp+fp)



def recall(tp,fp,fn,tn):
    """Fraction of all true positives achieved by model"""
    return tp/(tp+fn)

sensitivity=recall




def specificity(tp,fp,fn,tn):
    """Fraction of all true positives achieved by model"""
    return tn/(tn+fp)

def f1_score(tp,fp,fn,tn):
    p = precision(tp,fp,tn,fn)
    r = recall(tp,fp,tn,fn)
    return 2 * p * r/(p + r)

def rescue_code(function):
    """The best piece of code I've fucking ever seen"""
    import inspect
    get_ipython().set_next_input("".join(inspect.getsourcelines(function)[0]))

def predict(x_i, beta):
    return dot(x_i, beta)


def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)

def total_sum_of_squares(y):
    """the total squared variation of y_i's from their mean"""
    return sum(v ** 2 for v in de_mean(y))

def squared_error(x_i, y_i, beta):
    return error(x_i, y_i, beta) ** 2


def squared_error_gradient(x_i, y_i, beta):
    """the gradient corresponding to the ith squared error term"""
    return [-2 * x_ij * error(x_i, y_i, beta)
            for x_ij in x_i]

def estimate_beta(x, y):
    beta_initial = [random.random() for x_i in x[0]]
    return minimise_stochastic(squared_error, 
                               squared_error_gradient, 
                               x, y, 
                               beta_initial, 
                               0.001)  

def multiple_r_squared(x, y, beta):
    sum_of_squared_errors = sum(error(x_i, y_i, beta) ** 2 
                                for x_i, y_i in zip(x,y))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(y)

def bootstrap_sample(data):
    """randomly samples len(data) elements with replacement"""
    return [random.choice(data) for _ in data]
    
def bootstrap_statistic(data, stats_fn, num_samples):
    """evaluates stats_fn on num_samples bootstrap samples from data"""
    return [stats_fn(bootstrap_sample(data)) 
            for _ in range(num_samples)]

def estimate_sample_beta(sample):
    """sample is a list of pairs (x_i, y_i)"""
    x_sample, y_sample = zip(*sample) #magic unzipping trick
    return estimate_beta(x_sample, y_sample)

def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        #if the coefficient is positive we need to compute
        #twice the probability of seeing an even *larger* value
        return 2*(1-normal_cdf(beta_hat_j/sigma_hat_j))
    else:
        #otherwise twice the probability of seeing a smaller value
        return 2*(normal_cdf(beta_hat_j / sigma_hat_j))


def ridge_penalty(beta, alpha):
    #we usually don't penalise beta_0
    return alpha*dot(beta[1:], beta[1:])

def squared_error_ridge(x_i, y_i, beta, alpha):
    """Estimate erorr plus ridge penalty on beta"""
    return error(x_i, y_i, beta) ** 2 + ridge_penalty(beta, alpha)

def ridge_penalty_gradient(beta, alpha):
    """gradient of just the ridge penalty"""
    return [0] + [2 * alpha * beta_j for beta_j in beta[1:]]

def squared_error_ridge_gradient(x_i, y_i, beta, alpha):
    """the gradient corresponding to the ith squared error term 
    including the ridge penalty"""
    return vector_add(squared_error_gradient(x_i, y_i, beta),
                     ridge_penalty_gradient(beta, alpha))

def estimate_beta_ridge(x, y, alpha):
    """use gradient descent to fit a ridge regression with penalty alpha"""
    beta_initial = [random.random() for x_i in x[0]]
    return minimise_stochastic( partial(squared_error_ridge, alpha=alpha),
                                partial(squared_error_ridge_gradient, alpha=alpha), x, y, beta_initial, 0.001)

def bucketise(point, bucket_size):
    """floor the point to the next lower multiple of bucket_size"""
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points, bucket_size):
    """buckets the points and counts how many in each bucket"""
    return Counter(bucketise(point, bucket_size) for point in points)

def plot_histogram(points, bucket_size, title=""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()

def random_normal():
    """returns a random draw from a standard normal distribution"""
    return norm.ppf(random.random())

def scale(data_matrix):
    """returns the means and std devs of each column"""
    num_rows, num_cols = shape(data_matrix)
    means = [mean(get_column(data_matrix,j)) for j in range(num_cols)]
    stdevs = [standard_deviation(get_column(data_matrix,j)) for j in range(num_cols)]
    return means, stdevs

def rescale(data_matrix):
    """rescales the input data so that each col has mean 0 and stdev 1. Does nothing to cols with 0 stdev"""
    means, stdevs = scale(data_matrix)

    def rescaled(i,j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] - means[j])/stdevs[j]
        else:
            return data_matrix[i][j]
        
    num_rows, num_cols = shape(data_matrix)
    return make_matrix(num_rows, num_cols, rescaled)