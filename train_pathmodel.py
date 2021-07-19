import tensorflow as tf
import argparse
import json 
import os
import cv2
import numpy as np
from utils import (
    round_filters,
    round_repeats,
    efficientnet_params,
    get_model_params
)
from loss_yolact import YOLACTLoss
from backbone import Mask
from server import client_generator
from tensorflow.keras.callbacks import ModelCheckpoint
def gen(hwm, host, port,m):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    
    X, Y= tup
    batch_size=X.shape[0]
    
    for i in range(X.shape[1]):
    
     
      Y_t = Y[:, i]
        
        
      X_t = X[:, i]
      if i==0:
        rnn_state=np.zeros([batch_size,512])
        yolact_pred,rnn_statee=m.predict(x=[X_t,rnn_state])
      else:
        rnn_state=rnn_statee[:,110:]
        yolact_pred,rnn_statee=m.predict(x=[X_t,rnn_state])
      with tf.GradientTape() as tape:
        predictions1,prediction2=m([X_t,rnn_state])
        loc_loss, conf_loss, mask_loss, seg_loss,custom_loss, total_loss=YOLACTLoss(predictions1, predictions2, Y_t, cfg.model_params["num_classes"])
      gradients= tape.gradient(total_loss, model.trainable_variables)
      summ=[tf.reduce_sum(tf.abs(gradients[i])) for i in range(len(gradients))]
      sum_bce_grad=tf.reduce_sum(summ)
  
      print(sum_bce_grad)
        
      #print(i,rnn_state)
      yield [X_t,rnn_state], Y_t

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=30, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')

  args = parser.parse_args()
 
  model=Mask(2)
  input1=tf.keras.Input(shape=(550,550,3),name='img')
  
  input2=tf.keras.Input(shape=(512),name='rnn')
  output1,output2=model.call(input1,input2)
  model =tf.keras.Model(inputs=[input1,input2],outputs=[output1,output2])
#  criterion=YOLACTLoss(**cfg.loss_params)
  

  
  
  #print(x.shape,y.shape)
  #model.summary()
  #gg=gen(20, args.host, port=args.port,m=model)
  model.compile(optimizer="adam", loss=YOLACTLoss)
  
 
  history=model.fit_generator(
    gen(20, args.host, port=args.port,m=model),
    steps_per_epoch=25623,
    epochs=args.epoch,
    validation_data=gen(20, args.host, port=args.val_port,m=model),
    validation_steps=5000,verbose=1)