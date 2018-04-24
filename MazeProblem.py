#Project by Rajat Sharma and Don Joseph
#Date : 22-04-2018

"""
Running the Code
1)Import the libraries matplotlib, tensorflow, keras, scikit-learn.
2)Loading the dataset - Training and Test dataset : change the variables
  data_path for Training dataset (where training dataset is stored) and test_data_path for Test dataset
  (where test dataset is stored).
3)Run the Python code.
"""

import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras import backend as K
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# Define data path from where the training dataset is stored
data_path = 'D:/FIT Study Material/Semester 2/Artificial Intelligence/Project - Maze Problem/Dataset/Training'
data_dir_list = os.listdir(data_path)

img_rows = 128
img_cols = 128
num_channel = 1
num_epoch = 100

# Define the number of classes
num_classes = 2

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of training dataset: ' + '{}'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (128, 128))
        img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print("\nImage data ", str(img_data.shape))

if num_channel == 1:
    img_data = np.expand_dims(img_data, axis=4)
    print(img_data.shape)

# Assigning labels
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

labels[0:2500] = 0
labels[2500:5000] = 1

names = ['Maze with path', 'Maze without path']

# Convert class labels to one-hot encoding
Y = utils.to_categorical(labels, num_classes)

# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Defining the model
input_shape = img_data[0].shape

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

# Viewing model configuration
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

# Training the model
print("Training the model")
hist = model.fit(X_train, y_train, batch_size=64, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))

# Visualizing losses and accuracy
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = range(num_epoch)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
plt.style.use(['classic'])

# Evaluating the model
print('\nEvaluating the model')
score = model.evaluate(X_test, y_test, verbose=0)  # show_accuracy=True,
print('\nTest Loss:', score[0])
print('Test accuracy:', score[1])

print('Predicting an image from the trained dataset:')
test_image = X_test[0:1]
if model.predict_classes(test_image) == 0:
    print(model.predict_classes(test_image), '--> Maze with solution path')
else:
    print(model.predict_classes(test_image), '--> Maze without solution path')

# Predicting images from the test dataset
test_data_path = 'D:/FIT Study Material/Semester 2/Artificial Intelligence/Project - Maze Problem/Dataset/Testing'

test_img_list = os.listdir(test_data_path)
for test_image in test_img_list:
    print('\nPredicting the images from the test dataset: ' + test_image)
    test_image = cv2.imread(test_data_path + '/' + test_image)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = cv2.resize(test_image, (128, 128))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255

    if num_channel == 1:
        test_image = np.expand_dims(test_image, axis=3)
        test_image = np.expand_dims(test_image, axis=0)
    else:
        test_image = np.expand_dims(test_image, axis=0)

    # Predicting the test image
    if model.predict_classes(test_image) == 0:
        print(model.predict_classes(test_image), '--> Maze with solution path')
    else:
        print(model.predict_classes(test_image), '--> Maze without solution path')

# Visualizing the intermediate layer
def get_featuremaps(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output, ])
    activations = get_activations([X_batch, 0])
    return activations


layer_num = 3
filter_num = 0

activations = get_featuremaps(model, int(layer_num), test_image)
feature_maps = activations[0][0]

fig = plt.figure(figsize=(16, 16))
plt.imshow(feature_maps[:, :, filter_num], cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num) + '.png')

num_of_featuremaps = feature_maps.shape[2]
fig = plt.figure(figsize=(16, 16))
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num = int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
    ax = fig.add_subplot(subplot_num, subplot_num, i + 1)
    ax.imshow(feature_maps[:, :, i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.png')

# Printing the confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
# y_pred = model.predict_classes(X_test)
target_names = ['class 0(Maze with solution path)', 'class 1(Maze without solution path)']

print('\n')
print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))
#print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    #This function prints and plots the confusion matrix.
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test, axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
plt.show()

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# Load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

model.save('model.hdf5')
loaded_model = load_model('model.hdf5')