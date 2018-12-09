import numpy as np
class deepnetwork(object):
    def __init__(self, x, y):
        self.x=x
        self.y=y
        self.W1=np.zeros(700,)#shape of weigths must match with shape of y
        self.W2=np.zeros(700,)
        self.W3=np.zeros(700,)
        
        
    #FORWARD WİTH 2 HİDDEN LAYERS
    def forward(self, x, bias=0.0):
        self.bias=bias
        """
        y=W*x + b this is as we now a linear regression which is the root of neural network
        Actually above equation is one layer network . to go forward we can set second layer
        as y1=W*y + b W:weight could be set as 0 or randomly but for this kind of basic ops.
        it is better set it as 0.
        """
        self.l1=self.relu(np.dot(self.W1, self.x) + bias)#you can change the activation function parameters to find best loss
        self.l2=self.relu(np.dot(self.W2, self.l1) + bias)
        o=np.dot(self.W3, self.l2) + bias
        return o
  
    
    #LOSS FUNCTİONS                  
    def loss_func(self, y):
        self.y=y
        return(np.mean(np.square(self.y - self.forward(x, 0.9))))
    def cross_ent_coss(self, y):#not sure if this work correctly
        o=self.forward(x, 0.9)
        return -1*np.sum(y*np.log(o))

    #ACTİVATİON FUNCTIONS    
    def softmax(self, A):
        return np.exp(A) / np.exp(A).sum() 
    def tanh(self, t):
        return np.tanh(t)
    def sigmoid(self, s):
        return 1/(1-np.exp(-s))
    def sigmoidprime(self, s):#for backpropagation
        return s*(1-s)
    def relu(self, x):
        return np.max([np.zeros(x.shape), x], axis=0)
                       
    #BACKWARD                   
    def backward(self, x, y, o):
        self.o_error = np.subtract(y, o)  # error in output
        self.o_delta = self.o_error *self.sigmoidprime(o)  
        self.z2_error = np.dot(self.o_delta, self.W2.T)  
        self.z2_delta = self.z2_error * self.sigmoidprime(self.l1) 
        self.z3_error=np.dot(self.z2_delta, self.W3.T)
        self.z3_delta=self.z3_error*self.sigmoidprime(self.l2)
        self.W1 += np.dot(self.x.T, self.z2_delta)  
        self.W2 += np.dot(self.l1.T, self.z3_delta) 
        self.W3 += np.dot(self.l2.T, self.o_delta)
    
    def train(self, x, y):
        pred = self.forward(x, 0.9)
        self.backward(x, y, pred)


x=abs(np.random.randn(700, 700))#we set our x's shape as square shape and y must have same dimension.
y=abs(np.random.randn(700,))          
dd=deepnetwork(x, y)
epochs=20 #to set the best epoch number you should find by try. As i understand for this algorithm it is about size of data
for i in range(epochs):
    loss=dd.loss_func(y)
    print("loss function:  ", loss)
    print("cross entropy loss: ", dd.cross_ent_coss(y))
    dd.train(x, y)
