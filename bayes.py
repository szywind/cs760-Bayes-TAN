import sys
import scipy.io.arff as sia
import numpy as np
import random
import math
import pylab
import re


def errMsg():
    print 'Usage: python bayes.py <train-set-file> <test-set-file> <n|t>'


    
def trainNB(classes, train):
    '''
    estimate the prior probabilities P(Yj) and conditional probabilities P(Xi|Yj)
    '''
    (N,D) = train.shape
    Xs = classes[:-1]
    Ys = classes[-1]   # the last one is the label
    nY = len(Ys)
    #assert(nY == 2)
    subtrain = [train[train[:,-1]==l,:] for l in Ys]
    
    #Cxy = [[[np.sum(subtrain[i][:,feat_id]==c) for i in xrange(nY)] for c in Xs[feat_id]] for feat_id in xrange(len(Xs))]
    Pxy = [[[(np.sum(subtrain[i][:,feat_id]==c)+1)/float(subtrain[i].shape[0]+len(Xs[feat_id])) for i in xrange(nY)] for c in Xs[feat_id]] for feat_id in xrange(len(Xs))]
    Py = [float(subtrain[i].shape[0]+1)/(N+nY) for i in xrange(nY)]
    '''
    # test code
    for i in Pxy:
        temp = np.array(i)
        print np.sum(temp,0)  
        print np.sum(temp,0)==np.array([1.,1.])
    for i in Cxy:
        temp = np.array(i)
        print np.sum(temp,0)  
        print np.sum(temp,0)==np.array([57,43])    
    '''
    return [Pxy, Py]
    
def testNB(P, test, classes, display=True):
    '''
    estimate postieria probability P(Yi|X)
    '''
    Pxy = P[0]
    Py = P[1]
    [Nt, D] = test.shape
    nY = len(classes[-1])
    #assert nY == 2
    logres = np.ones((Nt,nY))
    for i in xrange(Nt):
        for j in xrange(nY):
            sample = test[i,:]
            temp = 0
            for feat_id in xrange(D-1):
                #assert sample[feat_id] in classes[feat_id]
                class_id = classes[feat_id].index(sample[feat_id])
                temp += math.log(Pxy[feat_id][class_id][j])
            logres[i,j] = temp+ math.log(Py[j])
    res = np.exp(logres)
    res = res/np.sum(res,1).reshape(-1,1)
    
    ## output the prediction result
    if display:
        label_id = res.argmax(1)
        nCorrect = 0
        for i in xrange(len(test)):
            if classes[-1][label_id[i]] == test[i][-1]:
                nCorrect += 1
            print classes[-1][label_id[i]],' ', test[i][-1], ' ', res[i,label_id[i]]
        print
        print nCorrect
        return res
    else:
        label_id = res.argmax(1)
        nCorrect = 0
        for i in xrange(len(test)):
            if classes[-1][label_id[i]] == test[i][-1]:
                nCorrect += 1       
        return res, nCorrect

def myLog2(x):
    return math.log(x)/math.log(2)
    
def compWeight(classes, train):
    Xs = classes[:-1]
    Ys = classes[-1]   # the last one is the label
    nY = len(Ys)
    #assert(nY == 2)
    subtrain = [train[train[:,-1]==l,:] for l in Ys]
    
    nVertex = len(classes)-1
    W=np.random.rand(nVertex, nVertex)
    [Pxy, Py] = trainNB(classes, train)
    
    for i in range(nVertex):
        W[i][i] = -1.0
        for j in range(0, i):
            # conditional combined probability
            cpxxy = [[[(1.0+np.sum(np.logical_and(subtrain[k][:,i]==c, subtrain[k][:,j]==d)))/ \
            (subtrain[k].shape[0] + len(Xs[i])*len(Xs[j])) \
            for c in Xs[i]] for d in Xs[j]] for k in range(nY)]
            
            '''
            # test code
            for k in range(nY): 
                print sum(cpxxy[k])       # should be 1
            '''
            
            pxxy = [[[(1.0+np.sum(np.logical_and(subtrain[k][:,i]==c, subtrain[k][:,j]==d)))/ \
            (train.shape[0] + len(Xs[i])*len(Xs[j])*nY) \
            for c in Xs[i]] for d in Xs[j]] for k in range(nY)]
            '''
            # test code
            print sum(pxxy)  # should be 1
            '''
            
            temp = 0
            for ii in range(len(Xs[i])):
                for jj in range(len(Xs[j])):
                    for k in range(nY):
                        temp += pxxy[k][jj][ii]*myLog2(cpxxy[k][jj][ii]/(Pxy[i][ii][k]*Pxy[j][jj][k]))
            W[i][j] = W[j][i] = temp
    return W
                
def findMST(W):
    N = W.shape[0]
    y = [N]
    Vnew = {0:y}
    Enew = set()
    Vleft = range(1,N)
    while len(Vnew)<N:
        # print "Vleft = ",Vleft
        max_w = -1.1216
        best_edge = []
        for i in Vnew:
            for j in Vleft:
                if W[i][j] > max_w:
                    max_w = W[i][j]
                    best_edge = [i,j]
                elif W[i][j] == max_w:
                    if i<best_edge[0] or (i==best_edge[0] and j<best_edge[1]):
                        max_w = W[i][j]
                        best_edge = [i,j]
     
        # print best_edge 
        Vnew[best_edge[1]] = [best_edge[0]]+y
        Enew.add(tuple(best_edge))
        Vleft.remove(best_edge[1])  
               
    return [Vnew, Enew]          
 
def compProb(classes, train, x, CV):
    '''
    compute P(x | X)
    '''
    count = dict() 
    nCV = len(CV)
    if train.shape[0] == 0:
        if nCV > 0:  
            cv = CV[0]   
            for i in range(len(classes[cv])):      
                count[i] = compProb(classes, train, x, CV[1:]) 
            res = [cv, train.shape[0], count]
        else:
            for i in range(len(classes[x])):
                count[i] = float(1)/len(classes[x])
            res = [x, train.shape[0], count]        
    else:
        if nCV > 0:  
            cv = CV[0]   
            for i in range(len(classes[cv])): 
                subTrain = train[train[:,cv]==classes[cv][i],:]      
                count[i] = compProb(classes, subTrain, x, CV[1:]) 
            res = [cv, train.shape[0], count]
        else:
            for i in range(len(classes[x])):
                subTrain = train[train[:,x]==classes[x][i],:]      
                #count[i] = subTrain.shape[0] 
                count[i] = float(1+subTrain.shape[0])/(train.shape[0]+len(classes[x]))
            res = [x, train.shape[0], count]
    return res

def trainTAN(classes, train, Vnew):
    # compute conditional probability table
    CPT = dict()
    for i in Vnew:
        CPT[i] = compProb(classes, train, i, Vnew[i])
    return CPT

def lookupCPT(CPT, i, sample):
    x = CPT[i][0]
    temp = CPT[i]
    while 1:
        class_i = classes[x].index(sample[x])       
        temp = temp[-1][class_i]
        if x == i:
            return temp
        x = temp[0]
            
    
def testTAN(CPT, test, classes, display=True):
    [Nt, D] = test.shape
    nY = len(classes[-1])
    #assert nY == 2
    logres = np.ones((Nt,nY))
    
    for i in xrange(Nt):          
        for j in xrange(nY):
            sample_cp = list(test[i])    # NOTE: test[i] == test[i,:], deep copy!!!
            sample_cp[-1] = classes[-1][j]
            temp = 0
            for k in xrange(D):
                temp += math.log(lookupCPT(CPT, k, sample_cp))
            logres[i,j] = temp
            
    res = np.exp(logres)
    res = res/np.sum(res,1).reshape(-1,1) 
    
    ## output the prediction result
    if display:
        label_id = res.argmax(1)
        nCorrect = 0
        for i in xrange(len(test)):
            if classes[-1][label_id[i]] == test[i][-1]:
                nCorrect += 1
            print classes[-1][label_id[i]],' ', test[i][-1], ' ', res[i,label_id[i]]
        print
        print nCorrect
    
        return res
    else:
        label_id = res.argmax(1)
        nCorrect = 0
        for i in xrange(len(test)):
            if classes[-1][label_id[i]] == test[i][-1]:
                nCorrect += 1
    
        return res, nCorrect
        
def drawLearningCurves(classes, train, test, m, name, Nlist=[25,50,100], T=4):
    
    #random.seed(1216)
    
    if train.shape[0] < max(Nlist):
        print "Error: the number of samples cannot be larger than the number of training data"
        return
    
    acc = []  
    if m == 'n':
        '''
        Naive Bayes
        '''  
        title = "Naive Bayes"
        for n in Nlist:
            temp = 0
            for t in xrange(T):
                subtrain = train[ random.sample(range(train.shape[0]), n) ]
                
                ## estimate the prior probabilities P(Y) and conditional probabilities P(Xi|Y)
                P = trainNB(classes, subtrain)
                        
                ## test with Bayes Rule
                (Pyx, nCorrect) = testNB(P, test, classes, False)
                
                print "[%d]nCorrect = %d" %(n,nCorrect)
                temp += nCorrect
            
            temp = float(temp)/(T*test.shape[0])
            acc.append(temp)
         
            
    elif m == 't': 
        '''
        TAN: Tree Augmented Network
        ''' 
        title = "TAN"      
        for n in Nlist:
            temp = 0
            for t in xrange(T):
                subtrain = train[ random.sample(range(train.shape[0]), n) ]
                
                ## compute weight I(Xi, Xj | Y) for each possible edge (Xi, Xj) between features
                W = compWeight(classes, subtrain)
        
                ## find maximum weight spanning tree (MST) for the graph with Prim's algorithm and asign directions to the edges
                [Vnew, Enew] = findMST(W)
                
                # add node y
                Vnew[len(Vnew)] = [] 
                
                ## estimate the conditional probability table
                CPT = trainTAN(classes, subtrain, Vnew)
            
                ## test with Bayes Rule
                (Pyx, nCorrect) = testTAN(CPT, test, classes, False)
                
                print "[%d]nCorrect = %d" %(n,nCorrect)
                temp += nCorrect
                
            temp = float(temp)/(T*test.shape[0])
            acc.append(temp)
    else:
        print "Error: the input m should be 'n' or 't'."
        return
         
    # draw the learning curve                
    pylab.figure(1)
    pylab.plot(Nlist, acc, 'rx')
    pylab.plot(Nlist, acc, 'r', label = name) 
    pylab.title("Learning Curve of "+title +"\n n=[25,50,100]")
    pylab.xlabel("# of training samples")
    pylab.ylabel("average test-set accuracy")
    #pylab.legend(["minimum", "average", "maximum"], loc = "lower right")
    pylab.legend(loc = 'lower right')
    pylab.savefig('_'.join(["lc", name, m])+".jpg")
    pylab.show() 
    
    
    
args = [arg for arg in sys.argv]


try:
    trainFile = args[1]
    testFile = args[2] 
    
    name = re.sub("[^A-Za-z']+", ' ', trainFile)
    name = name[:name.find(' ')]
    
    m = args[3]   # 'n' stands for 'naive bayes', 't' stands for 'TAN'
       
    ## load training and test data
    trainData = sia.loadarff(trainFile)
    testData = sia.loadarff(testFile)  
    
    ## reshape the datasets
    train = np.array([[i for i in trainData[0][j]] for j in range(trainData[0].shape[0])])
    test = np.array([[i for i in testData[0][j]] for j in range(testData[0].shape[0])])

    ## get the feature names and the class names
    feats = trainData[1].names()
    temp = [trainData[1][feat] for feat in trainData[1].names()]
    classes = [ line[-1] for line in temp]
    labels = classes[-1]

    
    nFeats = len(feats)-1
    nLabels = len(labels)
    
    if m == 'n':
        '''
        Naive Bayes
        '''
        ## estimate the prior probabilities P(Y) and conditional probabilities P(Xi|Y)
        P = trainNB(classes, train)
        
        ## output the structure of Naive Bayes Net
        for i in xrange(len(feats)-1):
            print feats[i], ' ', feats[-1]
        print
            
        ## test with Bayes Rule
        Pyx = testNB(P, test, classes)
        
        ## draw learning curves
        drawLearningCurves(classes, train, test, m, name, Nlist=[25,50,100], T=4)

        
    elif m == 't': 
        '''
        TAN: Tree Augmented Network
        ''' 
           
        ## compute weight I(Xi, Xj | Y) for each possible edge (Xi, Xj) between features
        W = compWeight(classes, train)
        
        ## find maximum weight spanning tree (MST) for the graph with Prim's algorithm and asign directions to the edges
        [Vnew, Enew] = findMST(W)
        # add node y
        Vnew[len(Vnew)] = [] 
       
        ## estimate the conditional probability table
        CPT = trainTAN(classes, train, Vnew)
        
        ## output the structure of Naive Bayes Net
        for i in xrange(len(feats)-1):
            parent = [feats[j] for j in Vnew[i]]
            print feats[i], ' '.join(parent)
        print
  
        ## test with Bayes Rule
        Pyx = testTAN(CPT, test, classes)
        
        ## draw learning curves
        drawLearningCurves(classes, train, test, m, name, Nlist=[25,50,100], T=4)
        
    else:
        errMsg()
except:
    errMsg() 
       
"""


if 1:
    trainFile = './lymph_train.arff'
    testFile = './lymph_test.arff'
    m = 't'   # 'n' stands for 'naive bayes', 't' stands for 'TAN'
       
    ## load training and test data
    trainData = sia.loadarff(trainFile)
    testData = sia.loadarff(testFile)  
    
    ## reshape the datasets
    train = np.array([[i for i in trainData[0][j]] for j in range(trainData[0].shape[0])])
    test = np.array([[i for i in testData[0][j]] for j in range(testData[0].shape[0])])

    ## get the feature names and the class names
    feats = trainData[1].names()
    temp = [trainData[1][feat] for feat in trainData[1].names()]
    classes = [ line[-1] for line in temp]
    labels = classes[-1]

    
    nFeats = len(feats)-1
    nLabels = len(labels)
    
    if m == 'n':
        '''
        Naive Bayes
        '''
        ## estimate the prior probabilities P(Y) and conditional probabilities P(Xi|Y)
        P = trainNB(classes, train)
        
        ## output the structure of Naive Bayes Net
        for i in xrange(len(feats)-1):
            print feats[i], ' ', feats[-1]
        print
            
        ## test with Bayes Rule
        Pyx = testNB(P, test, classes)
        
        ## draw learning curves
        drawLearningCurves(classes, train, test, m, [25,50,100], T=4)

        
    elif m == 't': 
        '''
        TAN: Tree Augmented Network
        ''' 
           
        ## compute weight I(Xi, Xj | Y) for each possible edge (Xi, Xj) between features
        W = compWeight(classes, train)
        
        ## find maximum weight spanning tree (MST) for the graph with Prim's algorithm and asign directions to the edges
        [Vnew, Enew] = findMST(W)
        # add node y
        Vnew[len(Vnew)] = [] 
        print "1:"
       
        ## estimate the conditional probability table
        CPT = trainTAN(classes, train, Vnew)
        
        ## output the structure of Naive Bayes Net
        for i in xrange(len(feats)-1):
            parent = [feats[j] for j in Vnew[i]]
            print feats[i], ' '.join(parent)
        print
  
        ## test with Bayes Rule
        Pyx = testTAN(CPT, test, classes)
        
        ## draw learning curves
        drawLearningCurves(classes, train, test, m, [25,50,100], T=4)
        
    else:
        errMsg()
   
"""    