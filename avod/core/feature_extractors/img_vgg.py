def ImgVgg(inputs,scope='img_vgg'):
    """ Modified VGG for image feature extraction.

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
    vgg_config = self.config
    weight_decay= 0.0005
    with tf.variable_scope(scope, 'img_vgg', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        net = tf.keras.layers.Conv2D(filters = 32, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), 
                                     activation=tf.nn.relu, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv1")(inputs)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None)(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
        net = tf.keras.layers.Conv2D(filters = 32, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
                                     use_bias=False, padding='same')(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None)(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
            
        net = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=None, padding="valid")(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
            
        net = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same')(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None)(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
        net = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same')(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None)(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
            
        net = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=None, padding="valid")(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
            
        net = tf.keras.layers.Conv2D(filters = 128,kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same')(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None)(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
        net = tf.keras.layers.Conv2D(filters = 128, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
                                     use_bias=False, padding='same')(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None)(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
        net = tf.keras.layers.Conv2D(filters = 128, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same')(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None)(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
            
        net = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=None, padding="valid")(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
            
        net = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same') (net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None)(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
        net = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same')(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None)(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
        net = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same')(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None)(net)
        tf.compat.v1.add_to_collection("end_points_collection", net)
     
        
#         downsampling_factor = 8
#                     downsampled_shape = input_pixel_size / downsampling_factor

#                     upsampled_shape = \
#                         downsampled_shape * vgg_config.upsampling_multiplier

#                     feature_maps_out = tf.image.resize_bilinear(
#                         net, upsampled_shape)
        
        # Convert end_points_collection into a end_point dict.
        end_points = tf.compat.v1.get_collection("end_points_collection")
        #end_points = model.get_config()
        return net, end_points