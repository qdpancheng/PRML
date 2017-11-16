#-*- coding:utf-8 -*- 
'''
Created on 2017年11月6日

@author: pc

'''
import numpy as np
import math
import random

def ldf(x, y):
    'number of train dataset'
    N = x.shape[0]
    ret_cov = np.zeros((x.shape[1],x.shape[1]))
    num_label = np.unique(y)
    mean_value =[]
    cov = []
    P = []
    for item in num_label:
        index = np.argwhere(item == y)
        Ni = index.shape[0]
        P.append(Ni / N)
        s = 0
        'calculate the mean value'
        for ii in index:
            s += x[ii]
        mean_value.append(s / Ni)
        s = 0
        'calculate the covariance'
        for ii in index:
            s+=np.dot(np.transpose(x[ii]-mean_value[item]),x[ii]-mean_value[item])
        cov.append(s / Ni) 
        ret_cov += P[item]*cov[item]
    return mean_value, ret_cov, P


def qdf(x, y):
    'number of train dataset'
    N = x.shape[0]
    'number of class'
    num_label = np.unique(y)
    
    mean_value =[]
    cov = []
    P = []
    for item in num_label:
        index = np.argwhere(item == y)
        Ni = index.shape[0]
        P.append(Ni / N)
        s = 0
        'calculate the mean value'
        for ii in index:
            s += x[ii]
        mean_value.append(s / Ni)
        s = 0
        'calculate the covariance'
        for ii in index:
            s+=np.dot(np.transpose(x[ii]-mean_value[item]),x[ii]-mean_value[item])
        cov.append(s / Ni) 
    return mean_value, cov, P


def mqdf(x, y):
    'number of train dataset'
    N = x.shape[0]
    'number of class'
    num_label = np.unique(y)
    
    mean_value = []
    cov = []
    P = []
    eig_values = []
    eig_vectors = []
    for item in num_label:
        index = np.argwhere(item == y)
        Ni = index.shape[0]
        P.append(Ni / N)
        s = 0
        'calculate the mean value'
        for ii in index:
            s+=x[ii]
        mean_value.append(s / Ni)
        s = 0
        'calculate the covariance'
        for ii in index:
            s+=np.dot(np.transpose(x[ii]-mean_value[item]),x[ii]-mean_value[item])
        cov.append(s / Ni)
        (eig_value, eig_vector) = np.linalg.eig(cov[item])
        for ii in range(len(eig_value)):
            if eig_value[ii]==0:
                eig_value[ii] = 0.00001
            else:
                continue
        eig_values.append(eig_value)
        eig_vectors.append(eig_vector)

    return mean_value, P, eig_values, eig_vectors


def rdf():
    pass



class LDF(object):
    '''
                Linear discriminant function-LDF
    '''
    def __init__(self):
        pass
    
    def learn(self, train_x, train_y):
        '''
            build the model---train the parameter
        '''
        x = np.asarray(train_x)
        y = np.asarray(train_y) 
        'number of class'
        self.__classes = np.unique(y)
        self.__mean, self.__cov, self.__P = ldf(x, y)
    
    def test(self, test_x, test_y):
        '''
            use the test dataset to test the model
            return accuracy
        '''
        labels = self.__classes.shape[0]
        x = np.asarray(test_x)
        y = np.asarray(test_y)
        N = x.shape[0]
        
        if len(self.__cov) == np.linalg.matrix_rank(self.__cov):
            inv_cov = np.linalg.inv(self.__cov) 
        else:
            inv_cov = np.linalg.pinv(self.__cov)
        err = 0
        for item in range(x.shape[0]):
            g = []
            for ii in range(labels):  
                g.append(2*np.dot(np.dot(self.__mean[ii], inv_cov), np.transpose(x[item])) \
                         - np.dot(np.dot(self.__mean[ii], inv_cov), np.transpose(self.__mean[ii])) + 2*math.log(self.__P[ii]))
            if g.index(max(g)) == y[item]:
                continue
            else:
                err+=1       
        return 1 - err/N
    

class QDF(object):
    '''
        Quadratic discriminant function-QDF
    '''
    def __init__(self):
        pass
    
    def learn(self, train_x, train_y):
        '''
            build the model---train the parameter
        '''
        x = np.asarray(train_x)
        y = np.asarray(train_y) 
        'number of class'
        self.__classes = np.unique(y)
        self.__mean, self.__cov, self.__P = qdf(x,y)    
    
    
    def test(self, test_x, test_y):
        '''
            use the test dataset to test the model
            return accuracy
        '''
        labels = self.__classes.shape[0]
        x = np.asarray(test_x)
        y = np.asarray(test_y)
        N = x.shape[0]

        inv_cov = []
        det_cov = []
        for ii in range(len(self.__cov)):
            if len(self.__cov[ii]) == np.linalg.matrix_rank(self.__cov[ii]):
                inv_cov.append(np.linalg.inv(self.__cov[ii]))
                det_cov.append(math.log(np.linalg.det(self.__cov[ii])))
            else:
                inv_cov.append(np.linalg.pinv(self.__cov[ii]))
                det_cov.append(0.0001)
        err = 0
        for item in range(x.shape[0]):
            g = []
            for ii in range(labels): 
                g.append(-1 * np.dot(np.dot((x[item]-self.__mean[ii]), inv_cov[ii]), np.transpose((x[item]-self.__mean[ii]))) - det_cov[ii])
            if g.index(max(g)) == y[item]:
                continue
            else:
                err+=1       
        return 1 - err/N


class MQDF(object):
    '''
            Modified quadratic discriminant function-MQDF
    '''
    def __init__(self):
        pass

    def learn(self, train_x, train_y):
        '''
            build the model---train the parameter
        '''
        'number of class'
        self.__classes = np.unique(train_y)
        x = np.asarray(train_x)
        y = np.asarray(train_y)
        self.__mean, self.__P, self.__eval, self.__evec = mqdf(x,y)  
        d = x.shape[1]
        N = x.shape[0]
        ER = []
        sorted_eval = []
        sorted_evec = []
        
        'eigvalue   high------>low,'
        'eigvector  high------>low'
        for jj in range(self.__classes.shape[0]):
            sorted_index = np.argsort(self.__eval[jj])
            sorted_eval.append(self.__eval[jj][sorted_index[::-1]])
            sorted_evec.append(self.__evec[jj][:,sorted_index[::-1]])
            
        'hyper k'    
        for k in range(1,d-1):
            err = 0
            delta = []
            'calculate delta'
            for zz in range(self.__classes.shape[0]):
                delta.append(np.sum(sorted_eval[zz][k:d]))
            for ii in range(len(delta)):
                if delta[ii] < 1e-6 and delta[ii] > -1e-6:
                    delta[ii] = 0.0001
                else:
                    continue      
            for item in range(x.shape[0]):
                g = []
                for ii in range(self.__classes.shape[0]):   # different classes
                    proj_dis = np.dot(x[item]-self.__mean[ii], np.transpose(x[item]-self.__mean[ii]))
                    proj_dis = proj_dis - np.sum(np.square(np.dot(x[item]-self.__mean[ii], np.transpose(sorted_evec[ii][:,qq]))) for qq in range(0,k))
                    g.append(-1 * np.sum(1 / sorted_eval[ii][qq] * np.square(np.dot(x[item]-self.__mean[ii], np.transpose(sorted_evec[ii][:,qq]))) for qq in range(0,k)) \
                             -1 / delta[ii] * proj_dis - np.sum(math.log(sorted_eval[ii][qq]) for qq in range(0,k)) - (d - k) * math.log(delta[ii]))
    
                if g.index(max(g)) == y[item]:
                    continue
                else:
                    err+=1   
            ER.append(err / N)
        return ER.index(min(ER)) + 1
        
        
    
    def test(self, test_x, test_y, k):
        '''
            use the test dataset to test the model
             use different k to select the best accuracy
        '''
        labels = self.__classes.shape[0]
        x = np.asarray(test_x)
        y = np.asarray(test_y)
        d = x.shape[1]
        N = x.shape[0]
        sorted_eval = []
        sorted_evec = []
        'eigvalue   high------>low,'
        'eigvector  high------>low'
        for jj in range(labels):
            sorted_index = np.argsort(self.__eval[jj])
            sorted_eval.append(self.__eval[jj][sorted_index[::-1]])
            sorted_evec.append(self.__evec[jj][:,sorted_index[::-1]])
        
        err = 0
        delta = []
        'calculate delta'
        for zz in range(labels):
            delta.append(np.sum(sorted_eval[zz][k:d]))
                  
        for item in range(x.shape[0]):
            g = []
            for ii in range(labels):   # different classes
                proj_dis = np.dot(x[item]-self.__mean[ii], np.transpose(x[item]-self.__mean[ii]))
                proj_dis = proj_dis - np.sum(np.square(np.dot(x[item]-self.__mean[ii], np.transpose(sorted_evec[ii][:,qq]))) for qq in range(0,k))
                    
                g.append(-1 * np.sum(1 / sorted_eval[ii][qq] * np.square(np.dot(x[item]-self.__mean[ii], np.transpose(sorted_evec[ii][:,qq]))) for qq in range(0,k)) \
                        -1 / delta[ii] * proj_dis - np.sum(math.log(sorted_eval[ii][qq]) for qq in range(0,k)) - (d - k) * math.log(delta[ii]))
    
            if g.index(max(g)) == y[item]:
                continue
            else:
                err+=1   
        return 1 - (err/N)







class RDF(object):
    '''
        Regularized discriminant analysis-RDA
    '''
    def __init__(self):
        pass


    def learn(self, train_x, train_y):
        '''
            build the model---train the parameter
        '''
        'number of class'
        self.__classes = np.unique(train_y)
        x = np.asarray(train_x)
        y = np.asarray(train_y)
        self.__mean, self.__cov, self.__P = qdf(x,y)
        d = x.shape[1]
        delta2 = []
        SIGMA0 = np.zeros((d,d))

        for ii in range(self.__classes.shape[0]):
            delta2.append(1/d*np.trace(self.__cov[ii]))
            SIGMA0+=self.__P[ii] * self.__cov[ii]

        ER = []
        I = np.eye(d)
        'cross validation'
        for gamma in [0.1 * x for x in range(1,10)]:
            ERR1 = []
            for beta in [0.1*x for x in range(1,10)]:
                err = 0
                cov = []
                for jj in range(self.__classes.shape[0]):
                    cov.append((1-gamma)*((1-beta)*self.__cov[jj]+beta*SIGMA0)+gamma*delta2[jj]*I)
                inv_cov = np.linalg.inv(cov) 

                for item in range(x.shape[0]):    
                    g = []
                    for ii in range(self.__classes.shape[0]): 
                        g.append(-1 * np.dot(np.dot((x[item]-self.__mean[ii]), inv_cov[ii]), np.transpose((x[item]-self.__mean[ii]))) - math.log(np.linalg.det(cov[ii])))
                        
                    if g.index(max(g)) == y[item]:
                        continue
                    else:
                        err+=1            
                ERR1.append(err/x.shape[0])
            ER.append(ERR1)
        mi = ER[0][0]
        r = 0
        b = 0
        for i in range(len(ER)):
            for j in range(len(ER[0])):
                if mi > ER[i][j]:
                    r = i 
                    b = j 
                else:
                    continue
        return (r+1)/10, (b+1)/10
        
        
    
    def test(self, test_x, test_y, r, b):
        '''
            use the test dataset to test the model
        '''
        labels = self.__classes.shape[0]
        x = np.asarray(test_x)
        y = np.asarray(test_y)
        N = x.shape[0]
        d = x.shape[1]
        delta2 = []
        SIGMA0 = np.zeros((d,d))

        for ii in range(labels):
            delta2.append(1/d*np.trace(self.__cov[ii]))
            SIGMA0+=self.__P[ii] * self.__cov[ii]

        I = np.eye(d)
        
        err = 0
        cov = []
        for jj in range(self.__classes.shape[0]):
            cov.append((1-r)*((1-b)*self.__cov[jj]+b*SIGMA0)+r*delta2[jj]*I)
        inv_cov = np.linalg.inv(cov) 

        for item in range(x.shape[0]):    
            g = []
            for ii in range(self.__classes.shape[0]): 
                g.append(-1 * np.dot(np.dot((x[item]-self.__mean[ii]), inv_cov[ii]), np.transpose((x[item]-self.__mean[ii]))) - math.log(np.linalg.det(cov[ii])))
                        
            if g.index(max(g)) == y[item]:
                continue
            else:
                err+=1            
        return 1 - err/N



class Parzen_Window(object):
    '''
    
    '''
    def __init__(self):
        pass
    
    def randset(self, sample, label):
        '''
            random the dataset
        '''
        n = sample.shape[0]
        train_num = int(n*0.8)
        train_x=[]
        train_y=[]
        test_x=[]
        test_y=[]
        train_index = random.sample(range(0, n), train_num)
        for i in range(0, n):
            if i in train_index:
                train_x.append(sample[i])
                train_y.append(label[i])
            else:
                test_x.append(sample[i])
                test_y.append(label[i])
        return train_x, train_y, test_x, test_y
    
    def learn(self, train_x, train_y):
        '''
            build the model---train the parameter
        '''
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        self.__classes = np.unique(train_y)
        trn_x, trn_y, cv_x, cv_y = self.randset(train_x, train_y)
        trn_x = np.asarray(trn_x)
        trn_y = np.asarray(trn_y)
        cv_x = np.asarray(cv_x)
        cv_y = np.asarray(cv_y)
        d = trn_x.shape[1]
        P = []
        Ni = []
        N = trn_x.shape[0]
        labels = np.unique(train_y).shape[0]

        idx = []
        for ii in range(labels):
            index = np.argwhere(ii == trn_y)
            idx.append(index)
            Ni.append(index.shape[0]) 
            P.append(index.shape[0]/N)
        
        self.__index = idx
        self.__Ni = Ni
        self.__d = d
        
        ER = []
        for h in [0.25*w for w in range(1,25)]:
            err = 0
            for ii in range(cv_x.shape[0]):   #cv set
                g =[]
                for qq in range(labels):
                    pi = 0
                    for jj in idx[qq]:
                        pi += 1 / Ni[qq] * 1 / (np.power(np.sqrt(2*np.pi * np.square(h)),d)) * \
                                            np.exp(-1 * np.dot(cv_x[ii]-trn_x[jj],np.transpose(cv_x[ii]-trn_x[jj])) / 2 * np.square(h))

                    g.append(pi*P[qq])
                if g.index(max(g)) == cv_y[ii]:
                    continue
                else:
                    err+=1   
            ER.append(err/cv_x.shape[0])
        h = np.argmin(ER)

        return (h+1) * 0.25
        
        
        
        
        
    
    def test(self, train_x, train_y, test_x, test_y, hh):
        '''
            use the test dataset to test the model
        '''
        labels = self.__classes.shape[0]
        trn_x = np.asarray(train_x)
        trn_y = np.asarray(train_y)
        tst_x = np.asarray(test_x)
        tst_y = np.asarray(test_y)
        d = trn_x.shape[1]
        P = []
        Ni = []
        idx = []
        N = trn_x.shape[0]
        for ii in range(labels):
            index = np.argwhere(ii == trn_y)
            idx.append(index)
            Ni.append(index.shape[0]) 
            P.append(index.shape[0]/N)


        err = 0
        for ii in range(tst_x.shape[0]):   #cv set
            g =[]
            for qq in range(labels):
                pi = 0
                for jj in idx[qq]:
                    pi += 1 / Ni[qq] * 1 / (np.power(np.sqrt(2*np.pi * np.square(hh)),d)) * \
                            np.exp(-1 * np.dot(tst_x[ii]-trn_x[jj],np.transpose(tst_x[ii]-trn_x[jj])) / 2 * np.square(hh))

                g.append(pi*P[qq])
            if g.index(max(g)) == tst_y[ii]:
                continue
            else:
                err+=1
    
        return 1-err/tst_x.shape[0]
    
    
if __name__=='__main__':
    print('hello gaussian')
    
    
    
    