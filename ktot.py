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
from modell import EfficientNet,BiFPN
from detbackbone import EfficientDetBackbone
num_channels=112
conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }
bifpn=[BiFPN(num_channels,conv_channel_coef[2],first_time=True),BiFPN(num_channels,conv_channel_coef[2]),BiFPN(num_channels,conv_channel_coef[2])]
def h5_to_pb(h5_save_path,pb_save_name,block_args,global_params):
    if "json" not in h5_save_path:
       try:
          model=EfficientDetBackbone(2)
          input1=tf.keras.Input(shape=(768,768,3))
          output1,output2=model.call(input1)
          model=tf.keras.Model(inputs=[input1],outputs=[output1,output2])
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
   block_args,global_params=get_model_params('efficientnet-b2',None)
   parser=argparse.ArgumentParser(description='keras to pb')
   parser.add_argument('k',type=str,default="transition",help='keras file name')
   parser.add_argument('pb',type=str,default="transition",help='save pb name')
   #parser.add_argument('--json',type=bool,default=False,help='json format')
   args=parser.parse_args()
   h5_save_path=args.k
   h5_to_pb(h5_save_path,args.pb,block_args,global_params)
