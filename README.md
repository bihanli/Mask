# Here is a combination of YPgru and yolact.

~~~
dlc step:
step1:output h5
cd YPgru_YB
python backbone.py
step2:h5->pb
cd YPgru_YB
python ktot2.py Mask.h5 Mask.pb
step3:snpe
python ktot2.py Mask.h5 Mask.pb

https://drive.google.com/file/d/1FqurmH-ZhREuMhbUUA56rnwqUHclekYU/view?usp=sharing

~~~
~~~
training step:
step1:json to binary encoding
cd YPgru_YB
python ./data/coco_tfrecord_creator.py -train_image_dir '../../../yolact-master/data/coco/test' -val_image_dir '../../../yolact-master/data/coco/test' -train_annotations_file './data/coco/annotations/test_train.json' -val_annotations_file './data/coco/annotations/test_valid.json' -output_dir './data/coco/train'

step2:training
cd YPgru_YB
python train_model.py 
