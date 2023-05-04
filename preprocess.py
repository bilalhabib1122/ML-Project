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