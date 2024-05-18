import keras
from keras import layers

def load_dataset():    
    num_classes = 10
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()    
    x_train = X_train.reshape(60000, -1)
    x_test = X_test.reshape(10000, -1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = keras.utils.to_categorical(Y_train, num_classes)
    y_test = keras.utils.to_categorical(Y_test, num_classes)

    return x_train, y_train, x_test, y_test

def train_model():
    x_train, y_train, x_test, y_test = load_dataset()
    model = keras.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(784,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))
    model.save("mnist_model.keras")

if __name__ == "__main__":
    train_model()