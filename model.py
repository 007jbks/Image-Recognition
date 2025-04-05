import numpy as np
import pandas as pd
import tensorflow as tf
df = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')

X = df.drop(['label'],axis=1).values
X = X/255.0
X = X.reshape(-1,28,28,1)
y = df['label']
y = tf.keras.utils.to_categorical(y, num_classes=25)
print("Preprocessing Done")

train = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')
X_test = train.drop(['label'],axis=1).values
X_test=X_test/255.0
X_test = X_test.reshape(-1,28,28,1)
y_test = train['label']
y_test = tf.keras.utils.to_categorical(y_test,num_classes=25)
print("Initialising model")

def model():
    return tf.keras.Sequential([
        #First Layer
        tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D(pool_size=2,strides=2),

        #Dropout layer:
        tf.keras.layers.Dropout(0.4),
        #Second Layer
        tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
         #Dropout layer:
        tf.keras.layers.Dropout(0.4),
        #Third Layer
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        #Dropout(0.4)
        
        tf.keras.layers.Dropout(0.4),
        #classification layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(25,activation='softmax') #output layer
        
        
    ])
cnn = model()
print("Model initialised")

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X, y, epochs=10, batch_size=64, validation_split=0.1)
test_loss, test_acc = cnn.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

cnn.save("sign_model.h5")
