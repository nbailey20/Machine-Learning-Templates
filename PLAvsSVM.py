import random
from cvxopt import matrix
from cvxopt import solvers
import numpy as np

## Created by: Nick Bailey 1/2016

## Script that implements both PLA (Perceptron Learning Algorithm) and SVM (Support Vector Machine)
##   for approximating pseudo-randomly generated lines between [MIN, MAX]x[MIN, MAX] with N test points, 
##   and compares the results.
  

                        ##      CHANGE IF DESIRED
NUMTESTS = 75000        ### number of test points to evaluate to obtain statistically stable error data
REPEATNUM = 1000        ### number of times new data sets are created, algorithms run, and errors calculated


### Helper functions for Perceptron ----------------

def makeLine(a, b):     ## makes line between points a and b
    slope = float (b[1] - a[1]) / (b[0] - a[0])
    intercept = a[1] - slope * a[0]
    return [slope, intercept]



def evalFunc(function, x, y):       ## evaluates linear func = [slope, Yint] on (x,y)
    z = function[0] * x + function[1]
    if y > z:
        return 1
    elif z > y:
        return -1                   ## set 1 or -1 for classification purposes
    else:
        return 0


def clashes(points, guess):     ## determines which points are misclassified by guess
    answer = []                 ##   compared to actual target
    for i in points:
        temp = (guess[2] + guess[0]*i[0] + guess[1]*i[1])
        if temp > 0:
            temp = 1
        elif temp < 0:
            temp = -1
        if i[2] != temp:
            answer += [i]
    return answer

    ### -----------------------------------------------------


### Helper functions for Support Vector Machine -------------

solvers.options['show_progress'] = False  ## do not display quadratic solving progress



def genQuadCoeffs(TRAIN):       ## generates the quadratic coefficients matrix for CVXOPT QP solver
    LENGTH = len(TRAIN)
    temp = []
    for j in range(LENGTH):
        subA = []
        a = TRAIN[j][2]
        s = TRAIN[j][0]
        t = TRAIN[j][1]
        S = np.matrix([s,t])
        for k in range(LENGTH):
            x = TRAIN[k][0]
            y = TRAIN[k][1]
            b = TRAIN[k][2]
            X = np.matrix([x,y])
            meantime = (S*X.getT())
            meantime *= (a*b)
            meantime = meantime.tolist()
            subA += meantime[0]
        temp += [subA]

    P = matrix(temp, tc='d')  
    return P                              



    ### ------------------------------------------------------


## Perceptron Learning Algorithm ##
def perceptron(TRAIN):
    counter = 0                               ## counts number of iterations of PLA 
    probability = 0                           ## estimate of Eout
    testPoints = TRAIN

    HYPOTH = [0,0,0]                                    ## intialize hypothesis weights

    ## START PLA
    while len(clashes(testPoints, HYPOTH)) != 0:      
        counter += 1
        problems = clashes(testPoints, HYPOTH)
        j = random.randint(0, len(problems)-1)              ## correct random misclassified point 

        temp = (HYPOTH[2] + HYPOTH[0]*problems[j][0] + HYPOTH[1]*problems[j][1])
        if temp > 0:
            temp = 1
        elif temp < 0:
            temp = -1
        HYPOTH[0] += (problems[j][2]-temp) * problems[j][0]     ## upate weights
        HYPOTH[1] += (problems[j][2]-temp) * problems[j][1]
        HYPOTH[2] += (problems[j][2]-temp) * 1  
        ## END PLA             

    return HYPOTH






## Support Vector Machine Algorithm ##
def SVM(TRAIN):
    SV = 0
    error = 0
    LENGTH = len(TRAIN)
    THRESHOLD = 10**(-6)                        ## minimum value to determine if Xn is a SV

    lc = [-1.0 for i in range(LENGTH)]
    q = matrix(np.matrix(lc).getT())            ## matrix of -1's for linear term

    P = genQuadCoeffs(TRAIN)                    ## quadratic coefficients matrix

    tempB = []
    for l in range(LENGTH):
        subB = [0]*l + [-1] + [0]*(LENGTH-l-1)
        tempB += [subB]
    G = matrix(tempB, tc='d')                    ## matrix G for minimizing coeff inequal conditions 

    inequal = [0 for n in range(LENGTH)]
    h = matrix(inequal, tc='d')                     ## matrix h for minimizing inequal constraints

    a = []
    for z in range(LENGTH):
        a += [TRAIN[z][2]]
    A = matrix(np.matrix(a), tc='d')         ## matrix A for minimizing coeff equal condition                  

    b = matrix([0], tc='d')                             ## matrix b for minimizing equal constraint    


    sol = solvers.qp(P,q,G,h,A,b)['x']          ## obtain solution coeffs from CVXOPT QP 

    weights = [0, 0]
    for u in range(LENGTH):                                 ## get weights for HYPOTHESIS
        weights[0] += sol[u]*TRAIN[u][2]*TRAIN[u][0]
        weights[1] += sol[u]*TRAIN[u][2]*TRAIN[u][1]

    bSolver = []
    for r in range(LENGTH):                               ## calculate number of support vectors
        if sol[r] > THRESHOLD:
            bSolver += [r]
            SV += 1

    constant = [TRAIN[0][2] - (weights[0]*TRAIN[0][0] + weights[1]*TRAIN[0][1])] #calc constant weight

    answer = [SV] + [constant[0]] + weights
    return answer







## MAIN FUNCTION ## 
def Main(MIN, MAX, N):
    percentSVM = 0                              ## % of time SVM is better than PLA 
    supportVectors = 0                          ## avg number of support vectors per run
    totalErrorSV = 0
    totalErrorPLA = 0
    for q in range(REPEATNUM):
        a = random.uniform(MIN, MAX)
        b = random.uniform(MIN, MAX)
        c = random.uniform(MIN, MAX)
        d = random.uniform(MIN, MAX)
        pointA = [a, b]
        pointB = [c, d]
        LINE = makeLine(pointA, pointB)            ## create target function LINE

        BADSET = 1            ## make sure points are not all on one side of LINE
        while BADSET == 1:
            testPoints = []
            oldZ = 0         
            for i in range(N):
                x = random.uniform(MIN, MAX)
                y = random.uniform(MIN, MAX)
                z = evalFunc(LINE, x, y)
                if i == 0:
                    oldZ = z
                if oldZ != z:
                    BADSET = 0
                testPoints += [[x,y,z]]            ## create training set of N points

        infoPLA = perceptron(testPoints)           ## run algorithms on set
        infoSV = SVM(testPoints)

        errPLA = 0          ## holds error for PLA of current run
        errSVM = 0          ## holds error for SVM of current run

        ## evaluate Eout
        for k in range(NUMTESTS):                               
            g = random.uniform(MIN, MAX)            ## generate random point, use target to classify point
            h = random.uniform(MIN, MAX)
            truth = evalFunc(LINE, g, h)
            checkPLA = infoPLA[2] + infoPLA[0]*g + infoPLA[1]*h
            if checkPLA > 0:
                checkPLA = 1
            elif checkPLA < 0:
                checkPLA = -1

            checkSVN = infoSV[1] + infoSV[2]*g + infoSV[3]*h
            if checkSVN > 0:
                checkSVN = 1
            elif checkSVN < 0:
                checkSVN = -1           ## classify point based on hypothesis

            if truth != checkPLA:
                errPLA += 1

            if truth != checkSVN:       ## check both algorithms' classifications
                errSVM += 1

        if errSVM < errPLA:
            percentSVM += 1
        supportVectors += infoSV[0]     ## record which algorithm did better each iteration
        totalErrorSV += errSVM          ## also record overall error for both algorithms
        totalErrorPLA += errPLA

    avgSV = float (supportVectors) / 1000
    percentSVN = float (percentSVM) / 10

    print "Percent of time SVM is better than PLA:", percentSVN
    print "Average number of support vectors used in SVM: ", avgSV
    print 'Average Error of SVM: ', float (totalErrorSV) / 75000 / 1000
    print 'Average Error of PLA: ', float (totalErrorPLA) / 75000 / 1000






