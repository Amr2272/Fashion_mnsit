from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

def build_model():
    model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs=50):
    history=model.fit(x_train, y_train, epochs=epochs)
    return history


def evaluate(model,x_test,y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\n Test accuracy: {test_acc:.4f}", "Test loss : ",test_loss)

def predict(model,x_test):
    y_predicted = model.predict(x_test)
    return y_predicted

def label_prediction(prediction_array):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    predicted_class = prediction_array.argmax()
    return class_names[predicted_class]

def save_model(model, filename='model.keras'):
    model.save(filename)

def load_saved_model(filename='model.keras'):
    loaded_model = load_model(filename)
    return loaded_model

