import tensorflow as tf
# tf.nn.softmax_cross_entropy_with_logits

# tf.nn.softmax

# def jcs(x,y):
# return -1*tf.reduce_sum(y*tf.log(x)+(1-y)*tf.log(1-x))

# def jcs1(x,y):
# return (y*tf.log(x)+(1-y)*tf.log(1-x))


# def softmax1(x):
# return x/tf.reduce_sum(x)


x=tf.constant([[1.,200.,3.,4.],[4.,80.,9.,2.]],dtype=tf.float32)
y=tf.constant([[0.,1.,0.,0.],[0.,0.,0.,1.]],dtype=tf.float32)
a=y * tf.log(tf.nn.softmax(x))
a=tf.where(tf.is_nan(a),tf.ones_like(a)*0.0000001,y=a)
# y=y/10
b=tf.reduce_mean(-tf.reduce_sum(a,reduction_indices=[1]))
sess=tf.Session()
c=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x,labels=y))
print (sess.run([b,c,tf.nn.softmax(x),a]))


sess.close()




