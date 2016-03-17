import random
import math
import numpy as np
from scipy.optimize import curve_fit 

###		Created by: Nick Bailey 1/2016

###		Script that implements the Bias-Variance out of sample error method for approximating a target polynomial.
###			Generates NUMTRIALS number of random points two at a time (between MIN and MAX), 
###				interpolating the points using 5 provided sample polynomials, and averages overall all trials
###			 	to find an average Hypothesis function. 
###			The bias and variances terms of the overall error are provided for each sample polynomial. 



##			CHANGE THE SAMPLE POLYNOMIALS AND TARGET FUNCTION AS DESIRED IN fitEquation AND Main

NUMSAMPLEPOLYNOMIALS = 5



## Helper polynomial to pass to 
##   numpy curve fitting in fitEquation 
def func(x, a, b):
	return a*x**2 + b



## Helper function that creates polynomials of the desires form 'num'
##  by interpolating two provided points 'a,b'
def fitEquation(a, b, num):
    if num == 0:										## y = b form
        slope = 0
        intercept = (float (a[1]) + b[1]) / 2
    elif num == 1:										## y = ax form
        p = float (a[0]) * a[1]
        q = float (b[0]) * b[1]
        r = float (a[0]) * a[0]
        s = float (b[0]) * b[0]
        slope = (p+q) / (r+s)
        intercept = 0
    elif num == 2:										## y = ax + b form
        slope = (float (b[1]) - a[1]) / (b[0] - a[0])
        intercept = a[1] - float (slope) * a[0]
    elif num == 3:										## y = ax^2 form
        c = float (b[1])*b[0] + float (a[1])*a[0]
        d = float (a[0])*a[0]*a[0] + float (b[0])*b[0]*b[0]
        slope = c / d
        intercept = 0 
    elif num == 4:										## y = ax^2 + b form
        x = np.array([a[0], b[0]])
        y = np.array([a[1], b[1]])
        var, shit = curve_fit(func, x, y)		## ugly exact solution, use curve fitting instead
        slope = var[0]								## use equation described by method func
        intercept = var[1]

    return [slope, intercept]



## MAIN FUNCTION ##
## 
def Main(MIN, MAX, NUMTRIALS):
	for p in range(NUMSAMPLEPOLYNOMIALS):

		## create data set and hypothesis set
	    randLines = []
	    randPoints = []									
	    for i in range(NUMTRIALS):						## create two points each iteration
	        a = random.uniform(MIN, MAX)
	        b = math.sin(math.pi * a)				## default uses target function f(x) = sin(x), CHANGE AS DESIRED
	        c = random.uniform(MIN, MAX)
	        d = math.sin(math.pi * c)
	        pointA = [a, b]
	        pointB = [c, d]
	        HYPOTH = fitEquation(pointA, pointB, p)		## fit each kind of polynomial between newly generated points
	        randLines += [HYPOTH]
	        randPoints += [pointA] + [pointB]			## keep track of original points and polynomials between them
	 
	    ## average over all polynomials to obtain average hypothesis
	    avgHYPOTH = [0,0]									
	    for j in range(NUMTRIALS):
	    	avgHYPOTH[0] += randLines[j][0]
	    	avgHYPOTH[1] += randLines[j][1]
	    avgHYPOTH[0] = float (avgHYPOTH[0]) / NUMTRIALS
	    avgHYPOTH[1] = float (avgHYPOTH[1]) / NUMTRIALS
	    print "Average Hypothesis: " + avgHYPOTH

	    ## calculate BIAS
	    BIAS = 0										
	    for k in range(2*NUMTRIALS):			## we created two points each iteration 
	    	avgGuess = randPoints[k][0] * avgHYPOTH[0] + avgHYPOTH[1]
	    	if p == 3 or p == 4:		## 2nd degree polynomials have different form fromt the rest, CHANGE AS DESIRED
	    		avgGuess = randPoints[k][0]*randPoints[k][0]*avgHYPOTH[0] + avgHYPOTH[1]
	    	actual = randPoints[k][1]
	    	bError = (avgGuess - actual) * (avgGuess - actual)			## determine bias error for each point
	    	BIAS += bError	
	    BIAS = float (BIAS) / (2*NUMTRIALS)					## calculate average bias
	    print "Average Bias: " + BIAS

	    ## calculate VARIANCE
	    VARIANCE = 0									
	    for l in range(2*NUMTRIALS):				## evaluate our hypothesis at each point as above
	    	point = randPoints[l][0]
	    	tempError = 0
	    	check = avgHYPOTH[0] * point + avgHYPOTH[1]
	    	if p == 4 or p == 5:
	    		check = avgHYPOTH[0] * point * point + avgHYPOTH[1]
	    	for z in range(NUMTRIALS):
	    	    guess = randLines[z][0] * point + randLines[z][1]		## CHANGE BASED ON SAMPLE POLYNOMIALS
	    	    if p == 4 or p == 5:
	    	    	guess = randLines[z][0]*point*point + randLines[z][1]
	    	    vError = (guess - check) * (guess - check)			## calculate variance for each point
	    	    tempError += vError
	    	VARIANCE += tempError
	    VARIANCE = float (VARIANCE) / (2*NUMTRIALS) / NUMTRIALS		## calculate average variance
	    print  "Average Variance: " + VARIANCE
	    print



    

