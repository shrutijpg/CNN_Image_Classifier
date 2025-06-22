import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128,128)
BATCH_SIZE = 16 # at 1 time 16 images will go


#Data Augumentation and data loading approach
train_datagen = ImageDataGenerator(rescale = 1./255)#converting RGB image into the grey scale B&W
val_datagen = ImageDataGenerator(rescale = 1./255)

#loading,. 
train_data = train_datagen.flow_from_directory(
    'data/train',
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical'
)
val_data = val_datagen.flow_from_directory(
    'data/val',
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical'
)

num_class = len(train_data.class_indices)

model =models.Sequential([
    #very first layer
    layers.Conv2D(32 ,(3,3),activation = 'relu',input_shape=(128,128,3), padding='same'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64 ,(3,3),activation = 'relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128 ,(3,3),activation = 'relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64 ,(3,3),activation = 'relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(254 ,(3,3),activation = 'relu', padding='same'),
    layers.MaxPooling2D(2,2),
    

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(.3),
    layers.Dense(num_class,activation='softmax')
])

#have to find the loss function and then have to optimize it
model.compile(
    loss="categorical_crossentropy",
    optimizer = 'adam',
    metrics = ['accuracy']
)

#Now we will be able to call the model training
model.fit(
    train_data,
    epochs = 10,
    validation_data = val_data
)

model.save("cnn_classifier.h5")

print(model.summary())