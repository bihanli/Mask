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
from backbone import Mask
from server import client_generator
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.utils import plot_model
def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    
    X, Y = tup
    
    Y= Y[:, -1]
    
    if X.shape[1] == 1:  # no temporal context
      X = X[:, -1]
    
    yield X, Y

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
  os.environ["CUDA_VISIBLE_DEVICES"]="0" # 使用编号为1，2号的GPU
  
  
  
  input1=tf.keras.Input(shape=(550,550,3),name='img')
  input2=tf.keras.Input(shape=(512),name='rnn')
  model=Mask(2)
  output1,output2=model.call(input1,input2)
  #print(output)
  model =tf.keras.Model(inputs=[input1,input2],outputs=[output1,output2])
  filepath="./saved_model/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
  mode='min')
  callbacks_list = [checkpoint]
  model.summary()
  #plot_model(model ,to_file='model.png',show_shapes=True,show_dtype=True)
  #model.load_weights('./saved_model/eff2_fix1.keras')
  model.compile(optimizer="adam", loss="categorical_crossentropy")
  
  history=model.fit_generator(
    gen(20, args.host, port=args.port),
    #steps_per_epoch=25623,
    steps_per_epoch=2,
    epochs=args.epoch,
    validation_data=gen(20, args.host, port=args.val_port),
    #validation_steps=5000,verbose=1,callbacks=callbacks_list)
    validation_steps=5,verbose=1,callbacks=callbacks_list)
  np.save('./loss_history/loss',np.array(history.history['loss']))
  np.save('./loss_history/val_loss',np.array(history.history['val_loss']))
  print("Saving model weights and configuration file.")

  
  
  model.save_weights("./saved_model/Mask.keras", True)
  with open('./saved_model/Mask.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
  
