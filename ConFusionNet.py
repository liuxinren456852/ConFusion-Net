from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import numpy as np
import helper_tf_util
import time

# import tensorflow as tf
import tensorflow.compat.v1 as tf # choose tf version 1 for compatibility
tf.disable_v2_behavior()

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers

            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['color'] = flat_inputs[num_layers: 2 * num_layers] # add color
            self.inputs['neigh_idx'] = flat_inputs[2 * num_layers: 3 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[4 * num_layers:5 * num_layers]
            self.inputs['features'] = flat_inputs[5 * num_layers]
            self.inputs['labels'] = flat_inputs[5 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[5 * num_layers + 2]
            self.inputs['cloud_inds'] = flat_inputs[5 * num_layers + 3]

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name)
            self.Log_file = open(dataset.name + '_' + 'log_train_' + str(dataset.val_split) + '.txt', 'a')

        with tf.variable_scope('layers'):
            self.logits = self.inference(self.inputs, self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):
        ##### init #####
        d_out = self.config.d_out
        feature = inputs['features']
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0') # extract raw feature
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)	
        
        ##### Encoder #####
        f_encoder_list = [] # f_{E_l} feature list
        f_encoder_plus = [] # f_{E_l}^+ feature list
        f_encoder_minus = [] # f_{E_l}^- feature list
        for i in range(self.config.num_layers):
            	
            f_encoder_i = self.mixed_residual_expansion(feature, tf.concat([inputs['xyz'][i], inputs['color'][i]], axis=-1), inputs['neigh_idx'][i], d_out[i], 'Encoder_layer_' + str(i), is_training) # MRE
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i]) # RS
           	
            feature = f_sampled_i
            
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
            
            if i > 0:
                f_plus = helper_tf_util.conv2d_transpose(f_encoder_i, d_out[i-1], [1, 1], 'Encoder_plus_' + str(i), [1, 1], 'VALID', bn=True, is_training=is_training) # transconv
                f_encoder_plus.append(f_plus) # add f+
            
            if i < 4:
                f_minus = helper_tf_util.conv2d(f_sampled_i, d_out[i+1], [1, 1], 'Encoder_minus_' + str(i), [1, 1], 'VALID', True, is_training) # conv
                f_encoder_minus.append(f_minus) # add f-

        ##### Bottom #####
        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                 'decoder_0', [1, 1], 'VALID', True, is_training)
                                
        ##### Decoder #####
        f_decoder_list = [] # f_{D_l} feature list
        for i in range(self.config.num_layers):
            if i == 0: # first layer with -
                f_interp_0 = self.nearest_interpolation(feature, inputs['interp_idx'][-1]) # US
                f_sum_0 = tf.concat([f_encoder_minus[-1], f_interp_0], axis=3) # concat f- and f_D
                f_att_0 = self.attention_connection(f_encoder_list[-2], feature, inputs['interp_idx'][-1], is_training, 'layer_0') # enhancement via AC
                f_concat_0 = tf.concat([f_att_0, f_sum_0], axis=3)
                f_decoder_0 = helper_tf_util.conv2d_transpose(f_concat_0, f_encoder_list[-2].get_shape()[-1].value, [1, 1],
                                                       'Decoder_layer_0', [1, 1], 'VALID', bn=True, is_training=is_training) # plus f^
                feature = f_decoder_0 
                f_decoder_list.append(f_decoder_0) 
            
            if i > 0 and i < 4: # middle layers with - and +
                feature = tf.concat([feature, f_encoder_plus[-i]], axis=3) # concat f+ and f_D
                f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-i-1]) # US
                f_sum_i = tf.concat([f_encoder_minus[-i-1], f_interp_i], axis=3) # concat f-, f+ and f_D
                f_att_i = self.attention_connection(f_encoder_list[-i-2], f_decoder_list[-1], inputs['interp_idx'][-i-1], is_training, 'layer_'+ str(i)) # enhancement via AC
                f_concat_i = tf.concat([f_att_i, f_sum_i], axis=3)
                f_decoder_i = helper_tf_util.conv2d_transpose(f_concat_i, f_encoder_list[-i-2].get_shape()[-1].value, [1, 1], 
                                                       'Decoder_layer_' + str(i), [1, 1], 'VALID', bn=True, is_training=is_training) # plus f^
                feature = f_decoder_i
                f_decoder_list.append(f_decoder_i) 
            
            if i == 4: # last layer with +
                feature = tf.concat([feature, f_encoder_plus[-4]], axis=3) # concat f+ and f_D
                f_interp_4 = self.nearest_interpolation(feature, inputs['interp_idx'][-5]) # US
                f_att_4 = self.attention_connection(f_encoder_list[-6], f_decoder_list[-1], inputs['interp_idx'][-5], is_training, 'layer_4') # enhancement via AC
                f_concat_4 = tf.concat([f_att_4, f_interp_4], axis=3)
                f_decoder_4 = helper_tf_util.conv2d_transpose(f_concat_4, f_encoder_list[-6].get_shape()[-1].value, [1, 1],
                                                       'Decoder_layer_4', [1, 1], 'VALID', bn=True, is_training=is_training) # plus f^                                   
                feature = f_decoder_4
                f_decoder_list.append(f_decoder_4)
        
        ##### classifier #####
        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training) # fc
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training) # fc
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1') # dp
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                     is_training, activation_fn=None) # catagory
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out

    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    def mixed_residual_expansion(self, feature, xyzrgb, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training) # f_{E_l}^0
        
        f_pc = self.building_block(xyzrgb, f_pc, neigh_idx, d_out, name + 'AFA', is_training) # AFA x 2
        
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training, activation_fn=None) # MLP(f_{E_l}^1, f_{E_l}^2)
        
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID', activation_fn=None, bn=True, is_training=is_training) # MLP(f_{E_l})
        
        return tf.nn.leaky_relu(f_pc + shortcut)
                
    def building_block(self, xyzrgb, feature, neigh_idx, d_out, name, is_training):
        ### init ###
        d_in = feature.get_shape()[-1].value # get feature dim
        '''AFA_0'''
        # xyzrgb
        geo_xyzrgb = self.relative_position_encoding(xyzrgb, neigh_idx) # RPE
        f_xyzrgb = tf.nn.leaky_relu(tf.layers.batch_normalization(geo_xyzrgb, -1, 0.99, 1e-6, training=is_training)) # BN + ReLU
        # BiMLP (f_{P_i} := f_xyz)
        f_xyzrgb = helper_tf_util.conv2d(f_xyzrgb, d_in, [1, 1], name + 'Bimlp1', [1, 1], 'VALID', True, is_training,
                                      activation_fn=tf.nn.relu) # with BN and ReLU
        f_xyzrgb = helper_tf_util.conv2d(f_xyzrgb, d_in, [1, 1], name + 'Bimlp2', [1, 1], 'VALID', False, is_training,
                                      activation_fn=None) # without BN and ReLU
        # semantic feature                             
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        
        # offsets
        f_neigh_offset = helper_tf_util.conv2d(f_xyzrgb, f_neighbours.get_shape()[-1].value, [1, 1], name + 'feature_offset_1', [1, 1], 'VALID', True, is_training)
        f_xyzrgb_offset = helper_tf_util.conv2d(f_neighbours, f_xyzrgb.get_shape()[-1].value, [1, 1], name + 'xyzrgb_offset_1', [1, 1], 'VALID', True, is_training)
        f_geo = f_xyzrgb + f_xyzrgb_offset           
        f_sem = f_neighbours + f_neigh_offset    
        f_agg = tf.concat([f_geo, f_sem], axis=-1)
                 
        f_pc_agg1 = self.soft_pooling(f_agg, d_out // 2, name + 'soft_pooling_1', is_training) # f_{E_l}^1
        
        ''' AFA^1 '''
        # xyzrgb
        f_xyzrgb = tf.nn.leaky_relu(tf.layers.batch_normalization(f_xyzrgb, -1, 0.99, 1e-6, training=is_training)) # BN + ReLU       
        # BiMLP (f_{P_i} := f_xyz)
        f_xyzrgb = helper_tf_util.conv2d(f_xyzrgb, d_out // 2, [1, 1], name + 'Bimlp3', [1, 1], 'VALID', True, is_training,
                                      activation_fn=tf.nn.relu) # with BN and ReLU
        f_xyzrgb = helper_tf_util.conv2d(f_xyzrgb,d_out // 2, [1, 1], name + 'Bimlp4', [1, 1], 'VALID', False, is_training,
                                      activation_fn=None) # without BN and ReLU        
        # semantic feature                              
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg1, axis=2), neigh_idx)
        
        # offsets
        f_neigh_offset = helper_tf_util.conv2d(f_xyzrgb, f_neighbours.get_shape()[-1].value, [1, 1], name + 'feature_offset_2', [1, 1], 'VALID', True, is_training)
        f_xyzrgb_offset = helper_tf_util.conv2d(f_neighbours, f_xyzrgb.get_shape()[-1].value, [1, 1], name + 'xyzrgb_offset_2', [1, 1], 'VALID', True, is_training)
        
        # concat all features
        f_geo = f_xyzrgb + f_xyzrgb_offset           
        f_sem = f_neighbours + f_neigh_offset  
        f_agg = tf.concat([f_geo, f_sem], axis=-1)         
        f_pc_agg2 = self.soft_pooling(f_agg, d_out // 2, name + 'soft_pooling_2', is_training) # f_{E_l}^2
        
        f = tf.concat([f_pc_agg1, f_pc_agg2], axis=-1) # f_{E_l}^1 + f_{E_l}^2
        return f
    
    def relative_position_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature  
        
    def attention_connection(self, f_skip, f_decoder, interp_idx, is_training, name): # need interpolation
        d = f_skip.get_shape()[-1].value # init
        f_decoder = self.nearest_interpolation(f_decoder, interp_idx) # lower decoder need to US for alignment
        # get f_{Q_l}
        feature_q = helper_tf_util.conv2d(f_skip, d, [1, 1], name + 'fc1', [1, 1], 'VALID', False, is_training,
                                          activation_fn=None)
        # get f_{K_l}
        feature_k = helper_tf_util.conv2d(f_decoder, d, [1, 1], name + 'fc2', [1, 1], 'VALID', False,
                                               is_training, activation_fn=None)
        # get f_{V_l}
        feature_v = helper_tf_util.conv2d(f_decoder, d, [1, 1], name + 'fc3', [1, 1], 'VALID', False,
                                               is_training, activation_fn=None)

        f_feature = feature_v + f_skip # f_{V_l} + f_{E_l}

        f_corr = feature_q - feature_k + f_skip # f_{Q_l} - f_{K_l} + f_{E_l}

        # BiMLP
        f_corr = helper_tf_util.conv2d(f_corr, 2*d, [1, 1], name + 'fc4', [1, 1], 'VALID', True,
                                      is_training, activation_fn=tf.nn.relu)

        f_corr = helper_tf_util.conv2d(f_corr, d, [1, 1], name + 'fc5', [1, 1], 'VALID', False,
                                      is_training, activation_fn=None)
                                      
        W_e = tf.nn.softmax(f_corr, axis=-1)
        f_att_skip = tf.multiply(W_e, f_feature) # attention-enhanced
        
        return f_att_skip

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def soft_pooling(feature_set, d_out, name, is_training):
        # soft mask enhancement
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg
