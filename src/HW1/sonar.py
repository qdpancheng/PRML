'''
Created on 2017年11月12日

@author: pc
'''
import numpy as np
import random
import HW1.GaussianClassifiers as GC
import os



def ReadData(file_path):
    '''
        read the dataset and transform dataset to list
    '''
    sample=[]
    label=[]
    with open(file_path, 'r') as f:
        dataset = f.readlines()
        for line in dataset[0:]:
            line = line.rstrip('\n')
            line = line.split(',')
            data =[]
            for item in line[0:-1]:
                data.append(float(item))
            sample.append(data)
            label.append(line[-1])
    return sample, label


def label2num(label):
    '''
        change label to the number
    '''
    label_num = []
    d={}
    flag = 0
    for item in label:
        if item not in d.keys():
            d[item] = flag
            flag+=1
        else:
            continue            
    for item in label:
        label_num.append(d[item])
    return label_num

def randset(sample, label):
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


if __name__=='__main__':
    '1.read the dataset  1.1 get file path 1.2 open the file and read the data'
    file_path = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
    file_path += r'\DATASET\sonar\sonar.all-data'
    sample, label = ReadData(file_path)

    '2. change label to number and change list to array(np)'
    label = label2num(label)
    sample = np.asarray(sample)
    label = np.asarray(label)
    
    accuracy_LDF = []
    accuracy_QDF = []
    accuracy_MQDF = []
    hyper_k = []
    accuracy_RDA = []
    hyper_r = []
    hyper_b = []
    accuracy_PW = []
    hyper_h = []
    for i in range(10):
        '3. random select train and test from dataset'
        train_x, train_y, test_x, test_y = randset(sample, label)

        '4. Parametric classifier and non-parametric classifier'
        '4.1  LDF'
        classifier = GC.LDF()
        classifier.learn(train_x, train_y)
        accuracy_LDF.append(classifier.test(test_x, test_y)) 
        
        '4.2 QDF'
        classifier = GC.QDF()
        classifier.learn(train_x, train_y)
        accuracy_QDF.append(classifier.test(test_x, test_y))
        
        '4.3 MQDF'
        classifier = GC.MQDF()
        k = classifier.learn(train_x, train_y)
        aa = classifier.test(test_x, test_y, k)
        accuracy_MQDF.append(aa)
        hyper_k.append(k)
        
        '4.4 RDA'
        classifier = GC.RDF()
        r,b = classifier.learn(train_x, train_y)
        aa = classifier.test(test_x, test_y, r, b)
        accuracy_RDA.append(aa)
        hyper_b.append(b)
        hyper_r.append(r)
        
        '4.5 Parzen'
        classifier = GC.Parzen_Window()
        hh = classifier.learn(train_x, train_y)
        hyper_h.append(hh)
        aa = classifier.test(train_x, train_y, test_x, test_y, hh)
        accuracy_PW.append(aa)
        
        
    print('LDF accuracy is ',np.mean(np.asarray(accuracy_LDF)))
    print('QDF accuracy is ',np.mean(np.asarray(accuracy_QDF)))
    print('MQDF accuracy is ',np.mean(np.asarray(accuracy_MQDF)), 'k is', int(np.mean(np.asarray(hyper_k))))
    print('RDA accuracy is ',np.mean(np.asarray(accuracy_RDA)), 'r is', np.mean(np.asarray(hyper_r)), 'b is', np.mean(np.asarray(hyper_b)))
    print('Parzen Window accuracy is ',np.mean(np.asarray(accuracy_PW)), ' h is', np.mean(np.asarray(hyper_h)))














