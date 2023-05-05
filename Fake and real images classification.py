#Code for pre processing - Osama asad - 21025871
from tensorflow.keras.preprocessing.image import ImageDataGenerator
img_height = 64
img_width = 64
batch_size = 16

train_data_dir = './Group_Project_Data/Train'
validation_data_dir = './Group_Project_Data/Valid'
#adding augmentation on training data only
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True)
test_datagen = ImageDataGenerator(
    rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Code for model - Samir Patel - 21072112
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D

def build_cnn():
	model = Sequential()
	model.add(Conv2D(nb_filters1, (conv1_size, conv1_size), input_shape=(img_width, img_height, 3)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	model.add(Conv2D(nb_filters2, (conv2_size, conv2_size)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	model.add(Conv2D(nb_filters3, (conv3_size, conv3_size)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation("relu"))
	model.add(Dropout(0.4)) # increasing dropour ratio to combat overfitting
	model.add(Dense(classes_num, activation='softmax'))

	return model

# Code for training a model - Bilal Habib - 21083834
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from model import build_cnn
# for plotting loss and accuracy graphs
import matplotlib.pyplot as plt 

img_width, img_height = 64, 64
nb_train_samples = 6000
nb_validation_samples = 2000

# decreasing filters for simpler model
nb_filters1 = 8
nb_filters2 = 16
nb_filters3 = 32

conv1_size = 3
conv2_size = 3
conv3_size = 3

pool_size = 2
classes_num = 2
batch_size = 16
lr = 0.0001

train_data_dir = './Group_Project_Data/Train'
validation_data_dir = './Group_Project_Data/Valid'

model = build_cnn()
model.summary()

epochs=30 # decrease epochs to combat overfitting
sgd = optimizers.SGD(learning_rate = lr, momentum = 0.9, nesterov = False)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True)
test_datagen = ImageDataGenerator(
    rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

r = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size)

target_dir = './models3/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models3/galaxy_classifier_model.h5')
model.save_weights('./models3/galaxy_classifier_weights.h5')

model.save('./tf_models3')

plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(r.history['accuracy'])  
plt.plot(r.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(r.history['loss'])  
plt.plot(r.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()

# results on validation dataset
scores = model.evaluate_generator(validation_generator, steps=42)
print("Accuracy = ", scores[1])

# Code for prediction - 
