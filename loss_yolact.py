class YOLACTLoss(object):

    def __init__(self,
           loss_weight_cls=1,
           loss_weight_box=1.5,
           loss_weight_mask=6.125,
           loss_weight_seg=1,
           neg_pos_ratio=3,
           max_masks_for_train=100):
        self._loss_weight_cls = loss_weight_cls
        self._loss_weight_box = loss_weight_box
        self._loss_weight_mask = loss_weight_mask
        self._loss_weight_seg = loss_weight_seg
        self._neg_pos_ratio = neg_pos_ratio
        self._max_masks_for_train = max_masks_for_train

    def call(self, pred1, pred2, label, num_classes):
        # all prediction component
        pred_cls = pred1['pred_cls']
        pred_offset = pred1['pred_offset']
        pred_mask_coef = pred1['pred_mask_coef']
        proto_out = pred1['proto_out']
        seg = pred1['seg']

        # all label component
        cls_targets = label['cls_targets']
        box_targets = label['box_targets']
        positiveness = label['positiveness']
        bbox_norm = label['bbox_for_norm']
        masks = label['mask_target']
        max_id_for_anchors = label['max_id_for_anchors']
        classes = label['classes']
        num_obj = label['num_obj']

        # calculate num_pos
        loc_loss = self._loss_location(pred_offset, box_targets, positiveness) * self._loss_weight_box
        conf_loss = self._loss_class(pred_cls, cls_targets, num_classes, positiveness) * self._loss_weight_cls
        mask_loss = self._loss_mask(proto_out, pred_mask_coef, bbox_norm, masks, positiveness, max_id_for_anchors,
                                    max_masks_for_train=100) * self._loss_weight_mask
        seg_loss = self._loss_semantic_segmentation(seg, masks, classes, num_obj) * self._loss_weight_seg
        custom_loss=self._loss_custom(label,pred2)
        total_loss = loc_loss + conf_loss + mask_loss + seg_loss + custom_loss
        return loc_loss, conf_loss, mask_loss, seg_loss, custom_loss, total_loss

    def _loss_location(self, pred_offset, gt_offset, positiveness):

        positiveness = tf.expand_dims(positiveness, axis=-1)

        # get postive indices
        pos_indices = tf.where(positiveness == 1)
        pred_offset = tf.gather_nd(pred_offset, pos_indices[:, :-1])
        gt_offset = tf.gather_nd(gt_offset, pos_indices[:, :-1])

        # calculate the smoothL1(positive_pred, positive_gt) and return
        num_pos = tf.shape(gt_offset)[0]

        # calculate smoothL1 loss
        diff = tf.abs(gt_offset - pred_offset)
        less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
        l1loss = (less_than_one * 0.5 * diff ** 2) + (1.0 - less_than_one) * (diff - 0.5)
        loss_loc = l1loss / tf.cast(num_pos, l1loss.dtype)
        loss_loc = tf.reduce_sum(loss_loc)

        return loss_loc

    def _loss_class(self, pred_cls, gt_cls, num_cls, positiveness):

        # reshape pred_cls from [batch, num_anchor, num_cls] => [batch * num_anchor, num_cls]
        pred_cls = tf.reshape(pred_cls, [-1, num_cls])
        pred_cls_max = tf.reduce_max(tf.reduce_max(pred_cls, axis=-1))
        logsumexp_pred_cls = tf.math.log(
            tf.reduce_sum(tf.math.exp(pred_cls - pred_cls_max), -1)) + pred_cls_max - pred_cls[:, 0]

        logsumexp_pred_cls = tf.reshape(logsumexp_pred_cls, [tf.shape(gt_cls)[0], -1])
        non_neg_mask = tf.cast(tf.logical_not(gt_cls != 0), tf.float32)
        logsumexp_pred_cls = logsumexp_pred_cls * non_neg_mask

        idx = tf.argsort(logsumexp_pred_cls, axis=1, direction="DESCENDING")
        idx_rank = tf.argsort(idx, axis=1)

        num_pos = tf.expand_dims(
            tf.reduce_sum(tf.cast((positiveness == 1), tf.int32), axis=-1), axis=-1)
        num_neg = tf.clip_by_value(num_pos * self._neg_pos_ratio, clip_value_min=0,
                                   clip_value_max=tf.shape(positiveness)[-1] - 1)

        negative_bool = tf.broadcast_to((idx_rank < num_neg), tf.shape(idx_rank))
        negative_bool = tf.cast(negative_bool, logsumexp_pred_cls.dtype) * non_neg_mask

        idx_pos = tf.where(positiveness == 1)
        idx_neg = tf.where(negative_bool == 1)
        idxes = tf.concat([idx_pos, idx_neg], axis=0)

        pred_cls = tf.reshape(pred_cls, [-1, tf.shape(positiveness)[-1], num_cls])
        pred_selected = tf.gather_nd(pred_cls, idxes)
        gt_selected = tf.gather_nd(gt_cls, idxes)
        gt_selected = tf.one_hot(tf.cast(gt_selected, tf.int32), depth=num_cls)

        loss_conf = tf.nn.softmax_cross_entropy_with_logits(gt_selected, pred_selected)
        loss_conf = tf.reduce_sum(loss_conf) / tf.cast(tf.reduce_sum(num_pos), loss_conf.dtype)
        return loss_conf

    def _loss_mask(self, proto_output, pred_mask_coef, gt_bbox_norm, gt_masks, positiveness,
                   max_id_for_anchors, max_masks_for_train):

        shape_proto = tf.shape(proto_output)
        num_batch = shape_proto[0]
        proto_h = shape_proto[1]
        proto_w = shape_proto[2]
        loss_mask = 0.
        total_pos = 0

        for idx in tf.range(num_batch):
            # extract randomly postive sample in prejd_mask_coef, gt_cls, gt_offset according to positive_indices
            proto = proto_output[idx]
            mask_coef = pred_mask_coef[idx]
            mask_gt = gt_masks[idx]
            bbox_norm = gt_bbox_norm[idx]  # [100, 4] -> [num_obj, 4]
            pos = positiveness[idx]
            max_id = max_id_for_anchors[idx]

            pos_indices = tf.squeeze(tf.where(pos == 1))

            # If exceeds the number of masks for training, select a random subset
            old_num_pos = tf.size(pos_indices)
            # print("pos indices", pos_indices.shape)
            if old_num_pos > max_masks_for_train:
                perm = tf.random.shuffle(pos_indices)
                pos_indices = perm[:max_masks_for_train]

            pos_mask_coef = tf.gather(mask_coef, pos_indices)
            pos_max_id = tf.gather(max_id, pos_indices)

            # if only 1 positive or no positive
            if tf.size(pos_indices) == 1:
                pos_mask_coef = tf.expand_dims(pos_mask_coef, axis=0)
                pos_max_id = tf.expand_dims(pos_max_id, axis=0)
            elif tf.size(pos_indices) == 0:
                continue
            else:
                ...

            # [num_pos, k]
            gt = tf.gather(mask_gt, pos_max_id)
            bbox = tf.gather(bbox_norm, pos_max_id)
            # print(bbox[:5])
            num_pos = tf.size(pos_indices)
            # print('gt_me', gt.shape)
            total_pos += num_pos

            # [138, 138, num_pos]
            pred_mask = tf.linalg.matmul(proto, pos_mask_coef, transpose_a=False, transpose_b=True)
            pred_mask = tf.transpose(pred_mask, perm=(2, 0, 1))
            s = tf.nn.sigmoid_cross_entropy_with_logits(gt, pred_mask)
            s = utils.crop(s, bbox)

            # calculating loss for each mask coef correspond to each postitive anchor
            bbox_center = utils.map_to_center_form(tf.cast(bbox, tf.float32))
            area = bbox_center[:, -1] * bbox_center[:, -2]
            mask_loss = tf.reduce_sum(s, axis=[1, 2]) / area

            if old_num_pos > num_pos:
                mask_loss = mask_loss * tf.cast((old_num_pos / num_pos), mask_loss.dtype)
            loss_mask += tf.reduce_sum(mask_loss)

        return loss_mask / tf.cast(total_pos, loss_mask.dtype)

    def _loss_semantic_segmentation(self, pred_seg, mask_gt, classes, num_obj):

        shape_mask = tf.shape(mask_gt)
        num_batch = shape_mask[0]
        seg_shape = tf.shape(pred_seg)[1]
        loss_seg = 0.

        for idx in tf.range(num_batch):
            seg = pred_seg[idx]
            masks = mask_gt[idx]
            cls = classes[idx]
            objects = num_obj[idx]

            # seg shape (69, 69, num_cls)
            # resize masks from (100, 138, 138) to (100, 69, 69)
            masks = tf.expand_dims(masks, axis=-1)
            masks = tf.image.resize(masks, [seg_shape, seg_shape], method=tf.image.ResizeMethod.BILINEAR)
            masks = tf.cast((masks > 0.5), seg.dtype)
            masks = tf.squeeze(masks)

            # obj_mask shape (objects, 138, 138)
            obj_mask = masks[:objects]
            obj_cls = tf.expand_dims(cls[:objects], axis=-1)

            # create empty ground truth (138, 138, num_cls)
            seg_gt = tf.zeros_like(seg)
            seg_gt = tf.transpose(seg_gt, perm=(2, 0, 1))
            seg_gt = tf.tensor_scatter_nd_add(seg_gt, indices=obj_cls, updates=obj_mask)
            seg_gt = tf.transpose(seg_gt, perm=(1, 2, 0))
            loss_seg += tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(seg_gt, seg))
        loss_seg = loss_seg / tf.cast(seg_shape, pred_seg.dtype) ** 2 / tf.cast(num_batch, pred_seg.dtype)

        return loss_seg

    def _loss_custom(y_true, y_pred,delta=2.):
      batch_size=tf.shape(y_true)[0]
      valid_len=y_true[:,50]
      valid_len=tf.reshape(valid_len,[-1,1])
      true_valid_len=y_true[:,50]
      pred_valid_len=y_pred[:,100]
      true_std=tf.abs(y_true[:,:50]-y_pred[:,:50])
      pred_std=tf.math.softplus(y_pred[:,50:100])
      #valid_len_loss=tf.reduce_sum(y_true[:,51]-y_pred[:,101])/tf.cast(batch_size,dtype=tf.float32)
      true_points=y_true[:,:50]
      pred_points=y_pred[:,:50]
      true_lead=y_true[:,51:55]
  
      true_lead_prob=y_true[:,55]
      pred_lead=y_pred[:,101:108:2]
  
      pred_lead_prob=tf.keras.activations.sigmoid(y_pred[:,109])
      true_lead_std=tf.abs(true_lead-pred_lead)
  
      pred_lead_std=tf.math.softplus(y_pred[:,102:109:2])

      s=0.
      for i in range(batch_size):
    
          l=tf.cast(valid_len[i][0],dtype=tf.int32)
          #points huber loss 
          error_zero_point=tf.abs(pred_points[i,0.])
          error_zero_point_std=tf.abs(tf.subtract(pred_points[i,0],pred_std[i,0]))
          error_point=tf.abs(tf.subtract(true_points[i,:],pred_points[i,:]))
          error_std=tf.abs(tf.subtract(true_std[i,:],pred_std[i,:]))
          error_valid_len=tf.abs(tf.subtract(true_valid_len[i],pred_valid_len[i]))
          condition_zero_point=tf.less(error_zero_point,delta)
          condition_zero_point_std=tf.less(error_zero_point_std,delta)
          condition_point=tf.less(error_point,delta)
          condition_std=tf.less(error_std,delta)
          condition_valid_len=tf.less(error_valid_len,delta)
          small_res_point=0.5 * tf.square(error_point)  #mse
          small_res_std=0.5 * tf.square(error_std) #mse
          small_res_valid_len=0.5 * tf.square(error_valid_len)  #mse
          small_res_zero_point=0.5 * tf.square(error_zero_point) 
          small_res_zero_point_std=0.5 * tf.square(error_zero_point_std) 
          large_res_point= delta * error_point - 0.5 * tf.square(delta) 
          large_res_std= delta * error_std - 0.5 * tf.square(delta) 
          large_res_valid_len=delta * error_valid_len - 0.5 * tf.square(delta)
          large_res_zero_point= delta * error_zero_point - 0.5 * tf.square(delta) 
          large_res_zero_point_std= delta * error_zero_point_std - 0.5 * tf.square(delta) 
          point_loss=tf.where(condition_point,small_res_point,large_res_point)
          point_zero_point_loss=tf.where(condition_zero_point,small_res_zero_point,large_res_zero_point)
          point_zero_point_std_loss=tf.where(condition_zero_point_std,small_res_zero_point_std,large_res_zero_point_std)
          std_loss=tf.where(condition_std,small_res_std,large_res_std)
          valid_len_loss=tf.where(condition_valid_len,small_res_valid_len,large_res_valid_len)
          point_loss=tf.reduce_sum(point_loss[:l])
          std_loss=tf.reduce_sum(std_loss[:l])
          #points loss
          points_all_loss=tf.where(tf.equal(l,0),point_zero_point_loss,(point_loss)/tf.cast(l,dtype=tf.float32))
          points_std_all_loss=tf.where(tf.equal(l,0),point_zero_point_std_loss,(std_loss)/tf.cast(l,dtype=tf.float32))
          #lead huber loss
          error_lead=tf.abs(tf.subtract(true_lead[i,:],pred_lead[i,:]))
   
          error_lead_std=tf.abs(tf.subtract(true_lead_std[i,:],pred_lead_std[i,:]))
    
          condition_lead=tf.less(error_lead,delta)
    
          condition_lead_std=tf.less(error_lead_std,delta)
    
          small_res_lead=0.5 * tf.square(error_lead)
    
          small_res_lead_std=0.5 * tf.square(error_lead_std) 
    
          large_res_lead= delta * error_lead - 0.5 * tf.square(delta) 
  
          large_res_lead_std= delta * error_lead_std - 0.5 * tf.square(delta) 
    
          lead_loss=tf.where(condition_lead,small_res_lead,large_res_lead)
          lead_std_loss=tf.where(condition_lead_std,small_res_lead_std,large_res_lead_std)
    
          #lead loss
          lead_all_loss=tf.where(tf.equal(true_lead_prob[i],1),tf.reduce_sum(lead_loss)+tf.reduce_sum(lead_std_loss),0)/4. 
          lead_std_all_loss=tf.where(tf.equal(true_lead_prob[i],1),tf.reduce_sum(lead_std_loss),0)/4. 
          # BCE 
 
          true_prob=true_lead_prob[i]
          pred_prob=tf.maximum(tf.minimum(pred_lead_prob[i],1-1e-15),1e-15)
    
          bce=-1.*(true_prob*tf.math.log(pred_prob)+((1.-true_prob)*tf.math.log(1-pred_prob)))
          #s add all loss
          all_loss=points_all_loss+points_std_all_loss+lead_all_loss+lead_std_all_loss+bce+valid_len_loss
          s= s + all_loss
      loss_custom=s/tf.cast(batch_size,dtype=tf.float32)
  
      return loss_custom