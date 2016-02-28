import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
from scipy.special import expit
import math
from pylab import scatter, show, legend, xlabel, ylabel

plt.ion()

x1=[]
x2=[]
y=[]
for line in open('ex2data1.txt'):
    X,Y,Label = line.split(',')
    x1.append(float(X))
    x2.append(float(Y))
    y.append(int(Label))

x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)
num_samples=x1.shape[0]
ones = np.array(np.ones([num_samples,1]))

x1=x1.reshape(num_samples,1)
x2=x2.reshape(num_samples,1)

x = np.hstack([ones,x1,x2])

y=y.reshape(num_samples,1)
init_theta = np.array(np.zeros([x.shape[1],1]))

plt.scatter(x1,x2,c=y)
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Not admitted', 'Admitted'])


def sigmoid_function(theta_transpose_x):
    result = expit(theta_transpose_x)
    return result

def cost_function(init_theta,x,y):

    theta_transpose_x=x.dot(init_theta)
    positive_prob=float(np.dot((np.transpose(np.log(sigmoid_function(theta_transpose_x)))),y))
    negative_prob=float(np.dot((np.transpose(np.log(1-sigmoid_function(theta_transpose_x)))),(1-y)))
    cost_value=float((-positive_prob-negative_prob)/num_samples)
    gradient=np.dot(np.transpose(x),(sigmoid_function(theta_transpose_x)-y))/num_samples
    return cost_value


def plot(ind,x2):
    theta_list=ind
    x_axis=np.array([x2.min()-2,x2.max()+2])
    y_axis=(-1/ind[2])*(ind[0]+ind[1]*x_axis)
    y_axis=np.array(y_axis)
    plt.plot(x_axis,y_axis)

result=sp.fmin( cost_function, x0=init_theta, args=(x, y), maxiter=400, full_output=True,disp=True,retall=True)
final_theta=result[0]

plot(final_theta,x1)



new_sample_x,new_sample_y=input('Enter 2 values for prediction')
new_sample=np.array([1,new_sample_x,new_sample_y])
new_sample=new_sample.reshape(3,1)

print sigmoid_function(np.dot(final_theta,new_sample))
