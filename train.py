import glob
import pandas as pd
import argparse
import numpy as np ; print 'numpy ' + np.__version__
import tensorflow as tf ; print 'tensorflow ' + tf.__version__
import cv2 ; print 'cv2 ' + cv2.__version__

np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--width', help='ring size', default=256, type=int)
parser.add_argument('--model',help='tensorflow graph file',default='model.proto')
parser.add_argument('--lr', help='learning rate', default=0.00001, type=float)
parser.add_argument('--batch', help='batch size', default=1000, type=int)
parser.add_argument('--epochs', help='training epochs', default=1000000, type=int)
parser.add_argument('--n', help='synthetic training batches per epoch', default=1, type=int)
#parser.add_argument('--height', help='number of steps to simulate forward and backward', default=256, type=int)
parser.add_argument('--scale', help='scale factor for display', default=1, type=int)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()

def format_():
    #read in data
    path =r'/home/mattliston/fa/data' # use your path
    allFiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    frame.to_csv('log.csv')

    #collect dates
    data = frame.as_matrix()
    dates=[]
    for i in range(0,data.shape[0]):
        if data[i][0] not in dates:
            dates.append(data[i][0])

    #sort each date into its own dataframe
    sort = []
    for i in dates:
        sort.append(frame.loc[frame.Date==i])
    s = pd.concat(sort)
    s = s.as_matrix()


    #compress all information for each date into one line
    d=np.zeros((int(len(s)/len(allFiles))+1,len(allFiles)))    
    row=0
    column=0
    for i in range(0,s.shape[0]):
        if column==len(allFiles):
            row+=1
            column=0
        try:
            d[row][column]=float(s[i][5])/float(s[i][1])
        except (ValueError,TypeError):
            d[row][column]=np.random.randint(2) #bad data is randomised
        column+=1

    #quantize
    for i in range(0,d.shape[0]):
        for j in range(0,d.shape[1]):
            if np.isnan(d[i][j]):
                d[i][j]=np.random.randint(2)
            if d[i][j]>=1:
                d[i][j]=1
            if d[i][j]<1:
                d[i][j]=0
    
    return d.astype(int)
        
def genbatch(data,args):
    x=np.zeros((args.batch,data.shape[1]),dtype=int)
    y=np.zeros((args.batch,data.shape[1]),dtype=int)
    for i in range(0,args.batch):
        random = np.random.randint(data.shape[0]-1)
        x[i]=data[random]
        y[i]=data[random+1]
#    print x,y
    return x,y

#genbatch(format_(),args)
#exit(0)        


data=format_()        

x = tf.placeholder('float32', [None,data[0].shape[0]],name='x') ; print x
y = tf.placeholder('float32', [None,data[1].shape[0]],name='y') ; print y

n = tf.layers.conv1d(inputs=tf.expand_dims(x,-1),filters=128,kernel_size=8,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=128,kernel_size=4,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=128,kernel_size=2,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=128,kernel_size=1,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=128,kernel_size=1,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=128,kernel_size=1,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=128,kernel_size=1,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=64,kernel_size=1,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=32,kernel_size=2,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=16,kernel_size=4,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=8,kernel_size=4,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=4,kernel_size=4,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=1,kernel_size=8,padding='same',dilation_rate=1,activation=None) ; print n
#n = tf.layers.conv1d(inputs=n,filters=64,kernel_size=5,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
#n = tf.layers.conv1d(inputs=n,filters=32,kernel_size=7,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
#n = tf.layers.conv1d(inputs=n,filters=1,kernel_size=7,padding='same',dilation_rate=1,activation=None) ; print n

pred = tf.sigmoid(n)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(y,-1),logits=n)
opt = tf.train.AdamOptimizer(learning_rate=args.lr)
grads = opt.compute_gradients(loss)
train = opt.apply_gradients(grads)
norm = tf.global_norm([i[0] for i in grads])
init = tf.global_variables_initializer()

# draw blank visualization window and move it to a nice location on the screen
#fimg = np.zeros([args.height,args.width],dtype=np.uint8)
#bimg = np.zeros([args.height,args.width],dtype=np.uint8)
#img = np.zeros([2,data[0].shape[1]])
#cv2.imshow('vis',img,dsize=(0,0),fx=args.scale,fy=args.scale,interpolation=cv2.INTER_LANCZOS4)
#cv2.moveWindow('ca', 0,0)
#cv2.waitKey(10)

with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        # TRAIN
        larr=[] # losses
        narr=[] # gradients
        for j in range(args.n):
            y_,x_ = genbatch(data,args)
            _,l_,n_ = sess.run([train,loss,norm],feed_dict={x:x_,y:y_})
            larr.append(l_)
            narr.append(n_)

        # TEST
        y_,x_ = genbatch(data,args)
        p = np.squeeze(sess.run(pred,feed_dict={x:x_}))
        s = np.random.binomial(1,p)

        print 'epoch {:6d} loss {:12.8f} grad {:12.4f} accuracy {:12.8f}'.format(i,np.mean(larr),np.mean(narr),1-np.mean(np.abs(s-y_)))
        #print p[0,0:10]

        # VISUALIZE FORWARD AND BACKWARD
#        x_ = np.random.binomial(1,0.5,size=args.width)
#        for j in range(args.height):
#            fimg[j] = x_*255
#            x_ = rule110(x_)
#        for j in range(args.height-1,0,-1):
#            bimg[j] = x_*255
#            p = np.squeeze(sess.run(pred,feed_dict={x:[x_]}))
#            x_ = np.random.binomial(1,p)
#        cv2.imshow('vis',cv2.resize(np.concatenate([fimg,bimg],axis=1),dsize=(0,0),fx=args.scale,fy=args.scale,interpolation=cv2.INTER_LANCZOS4))
#        cv2.waitKey(10)






