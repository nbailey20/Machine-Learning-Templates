import random
import numpy as np
import math

## Creatd by: Nick Bailey 1/2016

##    Script that trains to recognize hand-written digits and evaluates performance for 2 types of classification.
##    This program takes In and Out of sample data from two files and performs Linear Regression
##    Also takes parameter k, where LAMBDA = 10^k: LAMBDA/N is the typical regularizer for regression

## File format:  [digit, intensity, symmetry] 
## TYPE: digit vs digit 'DVD', or digit vs all 'DVA'



## Helper functions ----------------

## helper to open and read input from files
def readFile(TYPE, filename, digitA, digitB):       
    result = []
    with open(filename) as f:
        content = f.readlines()

    for i in range(len(content)):                       ## split by spaces and convert all characters to floats
        temp = content[i].split()
        temp[0] = float (temp[0])
        temp[1] = float (temp[1])
        temp[2] = float (temp[2])

        if TYPE == 'DVD' and temp[0] == digitA:         ## set values to 1/-1 for classification purposes
            temp[0] = 1.0
            result += [temp]
        elif TYPE == 'DVD' and temp[0] == digitB:
            temp[0] = -1.0
            result += [temp]
        elif TYPE == 'DVA' and temp[0] == digitA:       ## different training/testing sets for different TYPEs
            temp[0] = 1.0
            result += [temp]
        elif TYPE == 'DVA' and temp[0] != digitA:
            temp[0] = -1.0
            result += [temp]

    return result     




## helper to approximate classification error:
## uses test set TEST, and determined weights 'w' to determine
##   if number is correctly classified
def evalError(TEST, w):                     
    error = 0  
    for i in range(len(TEST)):            
        s = TEST[i][1]
        t = TEST[i][2]
        actual = TEST[i][0]

        guess = w[0] + w[1]*s + w[2]*t + w[3]*s*t + w[4]*s*s + w[5]*t*t       ## CHANGE AS NEEDED

        if guess > 0:
            guess = 1.0
        elif guess < 0:
            guess = -1.0

        if actual != guess:
            error += 1

    return error



##----------------------------------




## MAIN FUNCTION ##
def linRegression(TYPE, fnameA, fnameB, digitA, digitB, k):
    weights = [0, 0, 0, 0, 0, 0]    ## initialized to all 0: CHANGE LENGTH AS NEEDED
    LAMBDA = 10**k

    ## correctly set values for classification type
    inTestPoints = readFile(TYPE, fnameA, digitA, digitB)
    outTestPoints = readFile(TYPE, fnameB, digitA, digitB)

    Xlist = []      ## declare lists to hold training points 
    Ylist = []          ## and their output

    for j in range(len(inTestPoints)):
        a = inTestPoints[j][1]
        b = inTestPoints[j][2]
        nonLinTransform = [1] + [a] + [b] + [a*b] + [a*a] + [b*b]      ## CHANGE AS NEEDED
        Xlist += [nonLinTransform]    ## of the form [1, x1, x2, x1x2, x1^2, x2^2]
        Ylist += [inTestPoints[j][0]]       ## actual classification result

    ## perform linear regression
    Xmatrix = np.matrix(Xlist)                         
    Ymatrix = np.matrix(Ylist)
    regMatrix = np.matrix(np.identity(len(weights)))        ## initialize regulizer matrix
    regMatrix = LAMBDA * regMatrix

    intermed = (Xmatrix.getT()*Xmatrix) + regMatrix
    intermed = intermed.getI()
    Xpseudo = intermed * Xmatrix.getT()

    Hypoth = Xpseudo * Ymatrix.getT()                ## obtain hypothesis weights from regression
    Hypoth = Hypoth.tolist()                                 

    for s in range(len(weights)):                    ## update weights with regression output
        weights[s] += Hypoth[s]

    ## calculate errors
    errorIN = evalError(inTestPoints, weights)       
    errorOUT = evalError(outTestPoints, weights)

    print 'Determined Weights: ', weights
    print "In-sample Error: ", float (errorIN) / len(inTestPoints)
    print "Out-of-sample Error: ", float (errorOUT) / len(outTestPoints)




    


