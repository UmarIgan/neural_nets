import numpy as np
import numpy.ma as mask
import matplotlib.pyplot as plt
from PIL import Image
import os
class deepnetwork(object):
    def __init__(self, x, y):
        self.x=x
        self.y=y
        #self.batch_size=batch_size
        #self.layers=layers
        #WEİGTHS
        self.W1=np.zeros(700,)
        self.W2=np.zeros(700,)
        self.W3=np.zeros(700,)
    #FORWARD WİTH 2 HİDDEN LAYERS
    def forward(self, x, bias=0.0, ):
        self.bias=bias
        
        self.l1=self.relu(np.dot(self.W1, self.x) + bias)
        self.l2=self.relu(np.dot(self.W2, self.l1) + bias)
        o=np.dot(self.W3, self.l2) + bias
        return o
  
    
    """def sgd(self, x, epochs, batch_size, eta, test_data=None):
        if test_data: n_test=len(test_data)
        n=len(x)
        for j in xrange(epochs):
           
            random.shuffle(training_data)
            mini_batches = [
                k[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print( "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)
    def update_mini_batch(self, mini_batch, eta):
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)"""
    #LOSS FUNCTİONS                  
    def loss_func(self, y):
        self.y=y
        return(np.mean(np.square(self.y - self.forward(x, 0.9))))
    def cross_ent_coss(self, y):
        o=self.forward(x, 0.9)
        return -1*np.sum(y*np.log(o))

                      

    
    #ACTİVATİON FUNCTIONS    
    def softmax(self, A):
        return np.exp(A) / np.exp(A).sum() 
        
    def tanh(self, t):
        return np.tanh(t)
    def sigmoid(self, s):
        return 1/(1-np.exp(-s))
    def sigmoidprime(self, s):
        return s*(1-s)
    def relu(self, x):
        return np.max([np.zeros(x.shape), x], axis=0)
                       
    #BACKWARD                   
    def backward(self, x, y, o):
        self.o_error = np.subtract(y, o)  # error in output
        self.o_delta = self.o_error *self.sigmoidprime(o)  # applying derivative of sigmoid to error
        self.z2_error = np.dot(self.o_delta, self.W2.T)  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * self.sigmoidprime(self.l1) # applying derivative of sigmoid to z2 error
        self.z3_error=np.dot(self.z2_delta, self.W3.T)
        self.z3_delta=self.z3_error*self.sigmoidprime(self.l2)
        self.W1 += np.dot(self.x.T, self.z2_delta)  # adjusting first set (input --> hidden) weights
        self.W2 += np.dot(self.l1.T, self.z3_delta) # adjusting second set (hidden --> output) weights
        self.W3 += np.dot(self.l2.T, self.o_delta)
    def train(self, x, y):
        
        o = self.forward(x, 0.9)
        self.backward(x, y, o)


x=abs(np.random.randn(700, 700))
y=abs(np.random.randn(700,))
#print(x, y)           
dd=deepnetwork(x, y)
#print(x.shape, y.shape)
"""for i in range(20):
    #print("actual output: ", y)
    #print("predicted output: ",str(dd.forward(x, 0.9)))
    loss=dd.loss_func(y)
    print("loss function:  ", loss)
    print("cross entropy loss: ", dd.cross_ent_coss(y))
    dd.train(x, y)"""

path=r'C:\Program Files\train_directory'
class img_to_data(object):
    def __init__(self, image_path, imges):
        self.image_path=image_path
        self.imges=imges
        
    def load(self, image_path, imges):
        self.imges=imges
        for i in os.listdir(image_path):
            imgs=plt.imread(os.path.join(image_path, i))
            if imgs is not None:
                imges.append(imgs)
     
    def show_image(self, imges):
        self.h=10
        self.w=10
        self.fig=plt.figure()
        self.columns=11
        self.rows=3
        for i in imges:
            print(i.shape)
        for i in range(1, self.columns*self.rows + 1):
            self.imges=imges#np.random.randint(10, size=(self.h, self.w))
            self.fig.add_subplot(i, 2, 1)
            self.fig.set_figheight(15)
            self.fig.set_width(15)
            plt.imshow(imges, aspect='equal' )
        plt.show()
    
    
    def maxpool(self, imges, f=2, s=2):
# https://github.com/Alescontrela/Numpy-CNN/blob/master/CNN/forward.py   
        self.imges=imges
        for image in imges:
            
            n_c, h_prev, w_prev = image.shape
            h = int((h_prev - f)/s)+1
            w = int((w_prev - f)/s)+1
    
            self.downsampled = np.zeros((n_c, h, w))
            for i in range(n_c):
#slide maxpool window over each part of the image and assign the max value at each step to the output
                self.curr_y = self.out_y = 0
                while curr_y + f <= h_prev:
                    self.curr_x = out_x = 0
                    while curr_x + f <= w_prev:
                        downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                        curr_x += s
                        out_x += 1
                    curr_y += s
                    out_y += 1
            return downsampled
            print(len(downsampled))
    def show_images(self, imges):
        self.l=int(len(self.imges)/2)
        plt.figure(figsize=(10, 10))
        for i in range(2*self.l):
            plt.subplot(2, 2*self.l, 2*i + 1)

            plt.imshow(imges[i])
            plt.axis('off')
        plt.show()
        print(imges)

imges=[]
drr=img_to_data(path, imges)
drr.load(path, imges)
#drr.show_images(imges)
drr.maxpool(imges, f=2, s=2)
#drr.dataa(imges)
#new_shape=drr.dataa(imges)            
#print(new_shape=


class mask_detection(object):
    def __init__(self, background, object_to_be_detect):
        self.background=background
        self.object_to_be_detect=object_to_be_detect
    def load(self, image_path, imges):
        self.imges=imges
        for i in os.listdir(image_path):
            imgs=plt.imread(os.path.join(image_path, i))
            if imgs is not None:
                imges.append(imgs)

    
def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "damage":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)



#https://github.com/mnielsen/neural-networks-and-deep-learning
