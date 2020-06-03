"""Contains modified VGG model definition to extract features from
Bird's eye view input.

Usage:
    outputs, end_points = BevVgg(inputs, layers_config)
"""

import tensorflow as tf

def BevVgg(inputs,scope='bev_vgg'):
    """ Modified VGG for BEV feature extraction

    Note: All the fully_connected layers have been transformed to conv2d
              layers and are implemented in the main model.

    Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False fo validation/testing.
            scope: Optional scope for the variables.

    Returns:
            The last op containing the log predictions and end_points dict.
    """
    def list_to_dict(lst):
        """Short function to convert the list of the output tensor's collection 
            into an indexed dictionary.
            
            Args: the collection list
            Returns: the dictionary from that list
            
        """
        op = { i : lst[i] for i in range(0, len(lst) ) }
        return op

    #vgg_config = self.config
    weight_decay = 0.0005
    
    shape=inputs.get_shape()
    input_pixel_size=shape[1]*shape[2]
                           
    with tf.compat.v1.variable_scope(scope, 'bev_vgg', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.

            
        net = tf.keras.layers.Conv2D(filters = 32, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), 
                                     activation=tf.nn.relu, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv1")(inputs)
        tf.compat.v1.add_to_collection("end_points_collection", net)
        net = tf.keras.layers.Conv2D(filters = 32, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
                                     use_bias=False, padding='same')(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
            
        net = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=None, padding="valid")(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
            
        net = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same')(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
        net = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same')(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
            
        net = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=None, padding="valid")(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
            
        net = tf.keras.layers.Conv2D(filters = 128,kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same')(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
        net = tf.keras.layers.Conv2D(filters = 128, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
                                     use_bias=False, padding='same')(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
        net = tf.keras.layers.Conv2D(filters = 128, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same')(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
            
        net = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=None, padding="valid")(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
            
        net = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same') (net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
        net = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same')(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
        net = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same')(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
     
        #model = tf.keras.models.Model(inputs=inputs, outputs=net)
        
        # Convert end_points_collection into a end_point dict [TODO, now just list]
        end_points = tf.compat.v1.get_collection("end_points_collection")
        #end_points = model.get_config()
        return net, end_points
