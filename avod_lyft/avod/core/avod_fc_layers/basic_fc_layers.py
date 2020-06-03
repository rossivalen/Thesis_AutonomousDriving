import tensorflow as tf
import tf_slim as slim

from avod.core.avod_fc_layers import avod_fc_layer_utils


def build(fc_layers_config,
          input_rois, input_weights,
          num_final_classes, box_rep,
          is_training,
          end_points_collection):

    """Builds basic layers

       Args:
           fc_layers_config: Fully connected layers config object
           input_rois: List of input roi feature maps
           input_weights: List of weights for each input e.g. [1.0, 1.0]
           num_final_classes: Final number of output classes, including
               'Background'
           box_rep: Box representation (e.g. 'box_3d', 'box_8c', 'box_4c')
           is_training: Whether the network is training or evaluating
           end_points_collection: End points collection to add entries to

       Returns:
           cls_logits: Output classification logits
           offsets: Output offsets
           angle_vectors: Output angle vectors (or None)
           end_points: End points dict
       """

    # Parse config
    fusion_method = fc_layers_config.fusion_method
    num_layers = fc_layers_config.num_layers
    layer_sizes = fc_layers_config.layer_sizes
    l2_weight_decay = fc_layers_config.l2_weight_decay
    keep_prob = fc_layers_config.keep_prob

    cls_logits, offsets, angle_vectors = _basic_fc_layers(num_layers=num_layers, layer_sizes=layer_sizes, input_rois=input_rois, 
                                                          input_weights=input_weights, fusion_method=fusion_method, l2_weight_decay=l2_weight_decay, 
                                                          keep_prob=keep_prob, num_final_classes=num_final_classes, box_rep=box_rep, is_training=is_training)

    end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    return cls_logits, offsets, angle_vectors, end_points


def build_output_layers(tensor_in,
                        num_final_classes,
                        box_rep,
                        output):
    """Builds flattened output layers

    Args:
        tensor_in: Input tensor
        num_final_classes: Final number of output classes, including
            'Background'
        box_rep: Box representation (e.g. 'box_3d', 'box_8c', 'box_4c')

    Returns:
        Output layers
    """
    layer_out = None
    input_tf = tf.keras.layers.Input(shape=None, tensor=tensor_in)
    if output == 'cls':
        # Classification
        layer_out = tf.keras.layers.Dense(units=num_final_classes, activation=None, scope='cls_out')(input_tf)
    elif output == 'off':
        # Offsets
        off_out_size = avod_fc_layer_utils.OFFSETS_OUTPUT_SIZE[box_rep]
        if off_out_size > 0:
            layer_out = tf.keras.layers.Dense(units=off_out_size, activation=None, scope='off_out')(input_tf)
        else:
            layer_out = None

    elif output == 'ang':
        # Angle Unit Vectors
        ang_out_size = avod_fc_layer_utils.ANG_VECS_OUTPUT_SIZE[box_rep]
        if ang_out_size > 0:
            layer_out = tf.keras.layers.Dense(units=ang_out_size, activation=None, scope='ang_out')(input_tf)
        else:
            layer_out = None

    return layer_out


def _basic_fc_layers(num_layers, layer_sizes,
                     input_rois, input_weights, fusion_method,
                     l2_weight_decay, keep_prob,
                     num_final_classes, box_rep,
                     is_training):

    if not num_layers == len(layer_sizes):
        raise ValueError('num_layers does not match length of layer_sizes')

    if l2_weight_decay > 0:
        weights_regularizer = tf.keras.regularizer.l2(l2_weight_decay)
    else:
        weights_regularizer = None

    # Feature fusion
    fused_features = avod_fc_layer_utils.feature_fusion(fusion_method, input_rois, input_weights)
    output_names = ['cls', 'off', 'ang']
    cls_logits = None
    offsets = None
    angles = None

    for output in output_names:
        # Flatten
        fc_drop = tf.keras.layers.Flatten(scope=output + '_flatten')(fused_features)
        for layer_idx in range(num_layers):
            fc_name_idx = 6 + layer_idx

            # Use conv2d instead of fully_connected layers.
            fc_layer = tf.keras.layers.Dense(layer_sizes[layer_idx], kernel_regularizer=weights_regularizer,
                                                scope=output + '_fc{}'.format(fc_name_idx))(fc_drop)
            fc_drop = tf.keras.layers.Dropout(rate=keep_prob, is_training=is_training,
                                              scope=output + '_fc{}_drop'.format(fc_name_idx))(fc_layer)
            fc_name_idx += 1
            
        if output == 'cls':
            cls_logits= build_output_layers(fc_drop,  num_final_classes, box_rep, output)
        elif output == 'off':
            offsets = build_output_layers(fc_drop, num_final_classes, box_rep, output)
        elif output == 'ang':
            angles = build_output_layers(fc_drop, num_final_classes, box_rep, output)

    return cls_logits, offsets, angles
