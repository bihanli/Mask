import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import argparse
import json 
from tensorflow.keras.models import model_from_json
import os
from utils import (
    round_filters,
    round_repeats,
    efficientnet_params,
    get_model_params
)
from backbone import EfficientNet,Ypgru,Mask

def h5_to_pb(h5_save_path,pb_save_name,compound_coef):
    if "json" not in h5_save_path:
       
       try:
         model = Mask(compound_coef)
         input1=tf.keras.Input(shape=(550,550,3))
         input2=tf.keras.Input(shape=(512),name='rnn_state')
         output1,output2=model.call(input1,input2)
         
         model=tf.keras.Model(inputs=[input1,input2],outputs=[output1,output2])
         model.load_weights('Mask.h5')
         model.summary()
       except ValueError:
              print("Error mode:You use 'json mode' to covert this model")
              return 1
    else:
       with open(h5_save_path,'r') as jfile:
            
            
            model=model_from_json(str(json.load(jfile)))
            if os.path.isfile(h5_save_path.replace('json','h5')):
          
               weights_file=h5_save_path.replace('json','h5')
               model.load_weights(weights_file)
            else:
               weights_file=h5_save_path.replace('json','keras')
               model.load_weights(weights_file)
               
    
    full_model = tf.function(lambda Input: model(Input))
  
    full_model = full_model.get_concrete_function([tf.TensorSpec(model.inputs[i].shape, model.inputs[i].dtype) for i in range(len(model.inputs))])
    

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    #frozen_func.graph.as_graph_def()
    print('a')
    print(frozen_func.graph)
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    #for layer in layers:
        #print(layer)
    #print((frozen_func.inputs[0].name))
    #with open('./snpecommand.txt','w') as f:
         #f.write('snpe-tensorflow-to-dlc '+
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name=pb_save_name,
                      as_text=False)
if __name__=="__main__":
   
   parser=argparse.ArgumentParser(description='keras to pb')
   parser.add_argument('k',type=str,default="transition",help='keras file name')
   parser.add_argument('pb',type=str,default="transition",help='save pb name')
   #parser.add_argument('--json',type=bool,default=False,help='json format')
   args=parser.parse_args()
   h5_save_path=args.k
   h5_to_pb(h5_save_path,args.pb,2)
