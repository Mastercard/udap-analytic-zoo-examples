import tensorflow as tf

__author__ = 'e047349'

def getKerasModel(u_limit,m_limit,u_output,m_output,hdfs_path):
    input = tf.keras.layers.Input(shape=(3,))

    user_select = input[:, 0]
    item_select = input[:, 1]

    userEmbedding = tf.keras.layers.Embedding(u_limit + 1, u_output)(user_select)
    itemEmbedding = tf.keras.layers.Embedding(m_limit + 1, m_output)(item_select)

    u_flatten = tf.keras.layers.Flatten()(userEmbedding)
    m_flatten = tf.keras.layers.Flatten()(itemEmbedding)

    latent = tf.keras.layers.concatenate([u_flatten, m_flatten])

    numEmbeddingOutput = u_output + m_output
    linear1 = tf.keras.layers.Dense(numEmbeddingOutput // 2, activation="relu")(latent)
    output = tf.keras.layers.Dense(2, activation="softmax")(linear1)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    model.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    model.summary()
    return model
