import tensorflow as tf

def create_model(
        input_dim,
        n_rnn_layers=2
    ):
    input_layer = tf.keras.layers.Input(
        (None, input_dim), 
        dtype=tf.float32, 
        name='input_layer'
    )
    x = tf.keras.layers.Reshape(
        (-1, input_dim, 1),
        name='expand_dim_layer'
    )(input_layer)
    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        strides=2,
        activation='relu',
        name='conv_1'
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name='conv_1_bn'
    )(x)
    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        strides=2,
        activation='relu',
        name='conv_2'
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name='conv_2_bn'
    )(x)
    x = tf.keras.layers.Reshape(
        (-1, x.shape[-1] * x.shape[-2]),
        name='squeeze_layer'
    )(x)
    for idx in range(n_rnn_layers):
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                32,
                return_sequences=True
            ), 
            name=f'bilstm_{idx}'
        )(x) 

        if idx == n_rnn_layers - 1:
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    32,
                    return_sequences=False
                ),
                name=f'bilstm_{idx+1}'
            )(x) 
            x = tf.keras.layers.Dropout(
                0.2, 
                name='dropout_01'
            )(x)
    x = tf.keras.layers.Dense(
        64, 
        activation='relu',
        name="dense_1"
    )(x)
    x = tf.keras.layers.Dropout(
        0.3, 
        name='dropout_dense_01'
    )(x)
    output_layer = tf.keras.layers.Dense(
        2,
        activation='softmax',
        name='output_layer'
    )(x)

    model = tf.keras.Model(input_layer, output_layer)

    return model