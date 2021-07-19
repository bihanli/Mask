import tensorflow as tf
import config as cfg
from anchor import Anchor
from detection import Detect
class PredictionModule(tf.keras.layers.Layer):

  def __init__(self, out_channels, num_anchors, num_class, num_mask):
      super(PredictionModule, self).__init__()
      self.num_anchors = num_anchors
      self.num_class = num_class
      self.num_mask = num_mask

      self.Conv = tf.keras.layers.Conv2D(out_channels, (3, 3), 1, padding="same",
                        kernel_initializer=tf.keras.initializers.glorot_uniform(),
                        activation="relu")

      self.classConv = tf.keras.layers.Conv2D(self.num_class * self.num_anchors, (3, 3), 1, padding="same",
                           kernel_initializer=tf.keras.initializers.glorot_uniform())
      self.boxConv = tf.keras.layers.Conv2D(4 * self.num_anchors, (3, 3), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform())
      # activation of mask coef is tanh
      self.maskConv = tf.keras.layers.Conv2D(self.num_mask * self.num_anchors, (3, 3), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform())

  def call(self, p):
      p = self.Conv(p)
      pred_class = self.classConv(p)
      pred_box = self.boxConv(p)
      pred_mask = self.maskConv(p)

      # pytorch input  (N,Cin,Hin,Win) 
      # tf input (N,Hin,Win,Cin) 
      # so no need to transpose like (0, 2, 3, 1) as in original yolact code
      # reshape the prediction head result for following loss calculation
      pred_class = tf.reshape(pred_class, [tf.shape(pred_class)[0], -1, self.num_class])
      pred_box = tf.reshape(pred_box, [tf.shape(pred_box)[0], -1, 4])
      pred_mask = tf.reshape(pred_mask, [tf.shape(pred_mask)[0], -1, self.num_mask])

      # add activation for conf and mask coef
      # pred_class = tf.nn.softmax(pred_class, axis=-1)
      pred_mask = tf.keras.activations.tanh(pred_mask)

      return pred_class, pred_box, pred_mask
class ProtoNet(tf.keras.layers.Layer):
  """
      Creating the component of ProtoNet
      Arguments:
      num_prototype
  """

  def __init__(self, num_prototype):
      super(ProtoNet, self).__init__()
      self.Conv1 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform(),
                          activation="relu")
      self.Conv2 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform(),
                          activation="relu")
      self.Conv3 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform(),
                          activation="relu")
      self.upSampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
      self.Conv4 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform(),
                          activation="relu")

      self.finalConv = tf.keras.layers.Conv2D(num_prototype, (1, 1), 1, padding="same",
                          kernel_initializer=tf.keras.initializers.glorot_uniform(),
                          activation='relu')

  def call(self, p3):
      # (3,3) convolution * 3
      proto = self.Conv1(p3)
      proto = self.Conv2(proto)
      proto = self.Conv3(proto)

      # upsampling + convolution
      proto = tf.keras.activations.relu(self.upSampling(proto))
      proto = self.Conv4(proto)
       # final convolution
      proto = self.finalConv(proto)
      return proto

class FPN(tf.keras.layers.Layer):
    def __init__(self, num_fpn_filters):
        super(FPN, self).__init__()
        self.upSample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

        # no Relu for downsample layer
        self.downSample1 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.downSample2 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.lateralCov1 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.lateralCov2 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.lateralCov3 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())

        # predict layer for FPN
        self.predictP5 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                activation="relu")
        self.predictP4 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                activation="relu")
        self.predictP3 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                activation="relu")

    def call(self, c3, c4, c5):
        # lateral conv for c3 c4 c5
        p5 = self.lateralCov1(c5)
        p4 = self._crop_and_add(self.upSample(p5), self.lateralCov2(c4))
        p3 = self._crop_and_add(self.upSample(p4), self.lateralCov3(c3))
        # print("p3: ", p3.shape)

        # smooth pred layer for p3, p4, p5
        p3 = self.predictP3(p3)
        p4 = self.predictP4(p4)
        p5 = self.predictP5(p5)

        # downsample conv to get p6, p7
        p6 = self.downSample1(p5)
        p7 = self.downSample2(p6)

        return [p3, p4, p5, p6, p7]

    def _crop_and_add(self, x1, x2):
        """
        for p4, c4; p3, c3 to concatenate with matched shape
        https://tf-unet.readthedocs.io/en/latest/_modules/tf_unet/layers.html
        """
        x1_shape = x1.shape
        x2_shape = x2.shape
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        x1_crop = tf.cast(x1_crop, x2.dtype)
        return tf.add(x1_crop, x2)
    """
    def __init__(self,num_fpn_filters):
      super(FPN,self).__init__()
      self.upSample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

      # no Relu for downsample layer
      # Pytorch and tf differs in conv2d when stride > 1
      # https://dmolony3.github.io/Pytorch-to-Tensorflow.html
      # Hence, manually adding padding
      self.pad1 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
      self.downSample1 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="valid",
                            kernel_initializer=tf.keras.initializers.glorot_uniform())

      self.pad2 = tf.keras.layers.ZeroPadding2D(padding=(1,1))
      self.downSample2 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2, padding="valid",
                            kernel_initializer=tf.keras.initializers.glorot_uniform())

      self.lateralCov1 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                            kernel_initializer=tf.keras.initializers.glorot_uniform())
      self.lateralCov2 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                            kernel_initializer=tf.keras.initializers.glorot_uniform())
      self.lateralCov3 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding="same",
                            kernel_initializer=tf.keras.initializers.glorot_uniform())

      # predict layer for FPN
      self.predictP5 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                            activation="relu")
      self.predictP4 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                            activation="relu")
      self.predictP3 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding="same",
                             kernel_initializer=tf.keras.initializers.glorot_uniform(),
                             activation="relu")

    def call(self, c3, c4, c5):
      # lateral conv for c3 c4 c5
      # pytorch input  (N,Cin,Hin,Win) 
      # tf input (N,Hin,Win,Cin) 
      p5 = self.lateralCov1(c5)
      # _, h, w, _ = tf.shape(c4)
      p4 = tf.add(tf.image.resize(p5, [tf.shape(c4)[1],tf.shape(c4)[2]]), self.lateralCov2(c4))
      # _, h, w, _ = tf.shape(c3)
      p3 = tf.add(tf.image.resize(p4, [tf.shape(c3)[1],tf.shape(c3)[2]]), self.lateralCov3(c3))
      # print("p3: ", p3.shape)

      # smooth pred layer for p3, p4, p5
      p3 = self.predictP3(p3)
      p4 = self.predictP4(p4)
      p5 = self.predictP5(p5)

      # downsample conv to get p6, p7
      p6 = self.downSample1(self.pad1(p5))
      p7 = self.downSample2(self.pad2(p6))

      return [p3, p4, p5, p6, p7]
    """
class Yolact(tf.keras.Model):
  def __init__(self,backbone,fpn_channels,num_class,num_mask,anchor_params,detect_params):
    super(Yolact,self).__init__()
    self.backbone=backbone
    self.fpn_channels=fpn_channels
    self.num_class=num_class
    self.num_mask=num_mask
    self.anchor=anchor_params
    self.detect_params=detect_params
    self.protonet_coefficient=32
    #self.aspect_ratio=[1,0.5,2]
    #self.scale=[24,48,96,130,192]
    self._build()
  def _build(self):
    self.fpn=FPN(self.fpn_channels)
    self.protonet=ProtoNet(self.protonet_coefficient)
    # semantic segmentation branch to boost feature richness
    self.semantic_segmentation = tf.keras.layers.Conv2D(self.num_class-1, (1, 1), 1, padding="same",
                              kernel_initializer=tf.keras.initializers.glorot_uniform())
    self.num_anchors=Anchor(img_size=self.anchor["img_size"],feature_map_size=self.anchor["feature_map_size"],aspect_ratio=self.anchor["aspect_ratio"],scale=self.anchor["scale"])
    priors=self.num_anchors.get_anchors()
    self.pred=PredictionModule(self.fpn_channels,len(self.anchor["aspect_ratio"]),self.num_class,self.protonet_coefficient)
    #self.detect eval not train
    self.detect = Detect(anchors=priors, **cfg.detection_params)

    self.max_output_size = 300
    
  def call(self,feature_maps):
#(None, 80, 160, 16)(None, 40, 80, 24)(None, 20, 40, 48)(None, 10, 20, 120)(None, 5, 10, 352)    
    c3,c4,c5=feature_maps
    fpn_out=self.fpn.call(c3,c4,c5)
    p3=fpn_out[0]
    protonet_out = self.protonet.call(p3)
    # print("protonet: ", protonet_out.shape)

    # semantic segmentation branch
    seg = self.semantic_segmentation(p3)

    # Prediction Head branch
    pred_cls = []
    pred_offset = []
    pred_mask_coef = []

    # all output from FPN use same prediction head
    for f_map in fpn_out:
        cls, offset, coef = self.pred.call(f_map)
        pred_cls.append(cls)
        pred_offset.append(offset)
        pred_mask_coef.append(coef)
            
    pred_cls = tf.concat(pred_cls, axis=1)
    pred_offset = tf.concat(pred_offset, axis=1)
    pred_mask_coef = tf.concat(pred_mask_coef, axis=1)

    pred = {
            'pred_cls': pred_cls,
            'pred_offset': pred_offset,
            'pred_mask_coef': pred_mask_coef,
            'proto_out': protonet_out,
            'seg': seg
        }

    return pred
    """
    if training:
        pred = {
            'pred_cls': pred_cls,
            'pred_offset': pred_offset,
            'pred_mask_coef': pred_mask_coef,
            'proto_out': protonet_out,
            'seg': seg,
            'priors': self.priors
        }
        # Following to make both `if` and `else` return structure same
        result = {
            'detection_boxes': tf.zeros((self.max_output_size, 4)),
            'detection_classes': tf.zeros((self.max_output_size)), 
            'detection_scores': tf.zeros((self.max_output_size)), 
            'detection_masks': tf.zeros((self.max_output_size, 30, 30, 1)), 
            'num_detections': tf.constant(0)}
        pred.update(result)
    else:
        pred = {
            'pred_cls': pred_cls,
            'pred_offset': pred_offset,
            'pred_mask_coef': pred_mask_coef,
            'proto_out': protonet_out,
            'seg': seg,
            'priors': self.priors
        }

        pred.update(self.detect(pred))

    return pred
    """
  