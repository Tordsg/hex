import tensorflow as tf
import random


class ANET(tf.keras.Model):
    def __init__(self, output_size, layer_config):
        super(ANET, self).__init__()
        self.dense_layers = []
        for config in layer_config:
            neurons = config['neurons']
            activation = config['activation']
            self.dense_layers.append(tf.keras.layers.Dense(neurons, activation=activation))

        self.output_layer = tf.keras.layers.Dense(output_size, activation='softmax')

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)
    

def train(model, inputs, targets, epochs=1, batch_size=32):
    history = model.fit(inputs, targets, epochs=epochs, batch_size=batch_size, verbose=1)
    final_loss = history.history['loss'][-1]
    print("Final Loss:", final_loss)
    print("Final Accuracy:", history.history['accuracy'][-1])
    print("Final Precision:", history.history['precision'][-1])
    print("Final Recall:", history.history['recall'][-1])
    print("Final F1 Score:", history.history['f1_score'][-1])
    
def setup(ANET,learning_rate=0.01, optimizer='Adam'):
    if optimizer == 'Adagrad':
        opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'RMSProp':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ANET.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy', 'precision', 'recall', 'f1_score'])
    return ANET

def get_random_minibatch(inputs, targets, minibatch_size=32, metrics=['accuracy', 'precision', 'recall', 'f1_score']):
    if len(inputs) <= minibatch_size:
        return inputs, targets

    # Randomly select indices for the minibatch
    minibatch_indices = random.sample(range(len(inputs)), minibatch_size)

    # Extract minibatch inputs and targets using the selected indices
    minibatch_inputs = [inputs[i] for i in minibatch_indices]
    minibatch_targets = [targets[i] for i in minibatch_indices]

    return minibatch_inputs, minibatch_targets
