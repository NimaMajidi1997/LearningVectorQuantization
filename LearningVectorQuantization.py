import numpy as np
import matplotlib.pyplot as plt
import time
import random


def LVQ(var,S,alpha):
    a=var*np.random.randn(2,S)
    T_a=np.zeros((1,S))
    a[0,:]=a[0,:]-0.3
    a[1,:]=a[1,:]+1

    b=var*np.random.randn(2,S)
    T_b=1*np.ones((1,S))
    b[0,:]=b[0,:]-3.1
    b[1,:]=b[1,:]+1.3


    c=var*np.random.randn(2,S)
    T_c=2*np.ones((1,S))
    c[0,:]=c[0,:]-2.1
    c[1,:]=c[1,:]-1.7


    W1=np.array([a[:,random.randint(0,S-1)]]).T
    W2=np.array([b[:,random.randint(0,S-1)]]).T
    W3=np.array([c[:,random.randint(0,S-1)]]).T

    W=np.append(W1,W2,axis=1)
    W=np.append(W,W3,axis=1)
    W0=0
    W0+=W
    print('Identity of W0 is:',id(W0))
    print('Identity of W is:',id(W))
    print("\n")
    plt.plot(a[0,0:S-1],a[1,0:S-1],'go',markersize=4.5, label='a , T_0')
    plt.plot(b[0,0:S-1],b[1,0:S-1],'bo',markersize=4.5, label='b , T_1')
    plt.plot(c[0,0:S-1],c[1,0:S-1],'yo',markersize=4.5, label='c , T_2')

    plt.xlabel('X_1')
    plt.ylabel('X_2')


    plt.plot(W[0,0],W[1,0],'r',marker='*',markersize=11,label='Initial')
    plt.plot(W[0,1],W[1,1],'r',marker='*',markersize=11)
    plt.plot(W[0,2],W[1,2],'r',marker='*',markersize=11)


    X=np.append(a[:,0:S],b[:,0:S],axis=1)
    X=np.append(X[:,0:2*S],c[:,0:S],axis=1)

    T=np.append(T_a,T_b,axis=1)
    T=np.append(T,T_c,axis=1)

    per=list(np.random.permutation(T.shape[1]))
    X_per=X[:,per]
    T_per=T[0,per]
    CJ=0
    alpha=alpha
    d=np.zeros((1,3))
    tic=time.time()
    epoch=0
    Count=[3*S,3*S-1]
    W_temp=np.zeros((2,3))
    while Count[epoch+1]-Count[epoch] <= 0:
        count=0
        epoch+=1                         
        for i in range (0,3*S):
                            d[0,0]=np.sqrt(np.sum((X_per[:,i]-W[:,0])**2))
                            d[0,1]=np.sqrt(np.sum((X_per[:,i]-W[:,1])**2))
                            d[0,2]=np.sqrt(np.sum((X_per[:,i]-W[:,2])**2))

                            CJ = np.argmin(d)
                            W_temp=W
                            if CJ ==T_per[i]:
                                W[:,CJ]=  W[:,CJ]+alpha*(X[:,i]-W[:,CJ])
                               
                            else:
                                count+=1
                                W[:,CJ]=  W[:,CJ]-alpha*(X[:,i]-W[:,CJ])
        print('In epoch number=(%.0f) , number of misclassified samples=(%.0f)'%(epoch,count))
        Count.append(count)


    plt.plot(W_temp[0,0],W_temp[1,0],'k',marker='*',markersize=13,label='Final')
    plt.plot(W_temp[0,1],W_temp[1,1],'k',marker='*',markersize=13)
    plt.plot(W_temp[0,2],W_temp[1,2],'k',marker='*',markersize=13)
    plt.legend(loc='best', fontsize=16)
    plt.tight_layout()
    plt.show()
    toc=time.time()
    print("-----------------------------------------")
    print("Running time is: ",np.round((toc-tic)/60,2)," minutes")

    #Application

    Sample_C=0
    nerror_init=0
    nerror=0
    print("\n")
    print('W at initialization step is:',W0,sep='\n')
    print("-----------------------------------------")
    print('W after training  is:',W_temp,sep='\n')

    for i in range (0,3*S):
                            d[0,0]=np.sqrt(np.sum((X[:,i]-W_temp[:,0])**2))
                            d[0,1]=np.sqrt(np.sum((X[:,i]-W_temp[:,1])**2))
                            d[0,2]=np.sqrt(np.sum((X[:,i]-W_temp[:,2])**2))

                            CJ = np.argmin(d)
                            if CJ !=T[0,i]:
                                nerror+=1

                            
                            d[0,0]=np.sqrt(np.sum((X[:,i]-W0[:,0])**2))
                            d[0,1]=np.sqrt(np.sum((X[:,i]-W0[:,1])**2))
                            d[0,2]=np.sqrt(np.sum((X[:,i]-W0[:,2])**2))

                            CJ = np.argmin(d)
                            if CJ !=T[0,i]:
                                nerror_init +=1
                            Sample_C+=1
    print("\n")
    print('Error at the beginning is:',np.round(nerror_init/Sample_C*100,2),'%')
    print('Error at the end is:',np.round(nerror/Sample_C*100,2),'%')
    return

LVQ(var=1,S=100,alpha=0.0001)
