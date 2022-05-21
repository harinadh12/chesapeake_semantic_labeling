import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,\
    Dropout, BatchNormalization, Activation, Input, GlobalAveragePooling2D, UpSampling2D, Conv2DTranspose,Concatenate
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
import pickle
import numpy as np

def sequential_image_translator(input_shape,n_classes, conv_units, pool_size, lrate=0.001):
    """
    Create a sequential model for image translation
    ------------------------------------
    args:
        input_shape: tuple of ints (height, width, channels)
        n_classes: int
        lrate: float
        dropout: float
    ------------------------------------
    returns:
        model: keras.Model
    """

    model = Sequential()
    model.add(Input(shape=input_shape))
    for filters,pool in zip(conv_units,pool_size):
        model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
        if pool > 1:
            model.add(MaxPooling2D(pool_size=(2, 2)))

    for filters in conv_units[::-1]:
        model.add(Conv2DTranspose(filters, (3, 3), activation='relu',strides= (2,2), padding='same'))
        

   
    model.add(Conv2D(n_classes, (3,3), activation='softmax', padding='same'))

    opt = keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999, epsilon= None, decay=0.0, amsgrad= False)

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics='sparse_categorical_accuracy')
    plot_model(model, to_file='UNet_Sequential_API.png', show_shapes=True)
    print(model.summary())
    return model

def u_net_image_translator(input_shape, n_classes, conv_units, pool_size, lrate=0.0001):
    """
    Create a U-Net model for image translation
    ------------------------------------
    args:
        input_shape: tuple of ints (height, width, channels)
        n_classes: int  (number of classes)
        conv_units: list of ints (number of filters in each convolutional layer)
        pool_size: list of ints (size of the max pooling in each layer)
        lrate: float (learning rate)
    ------------------------------------
    returns:
        model: keras.Model
    """

    input_tensor = Input(shape=input_shape)
    x = input_tensor
    temp = []
    for filters,pool in zip(conv_units,pool_size):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        temp.append(x)

        if pool > 1:
            x = MaxPooling2D(pool_size=(2, 2))(x)
        
    for ind,filters in enumerate(conv_units[::-1]):
        x = Conv2DTranspose(filters, (3, 3), activation='relu', strides =(2,2), padding='same')(x)
        x = Concatenate()([x,temp[-ind-1]])
    
    output_tensor = Conv2D(n_classes,3, activation='softmax', padding='same')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    opt = keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999, epsilon= None, decay=0.0, amsgrad= False)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics='sparse_categorical_accuracy')
    plot_model(model, to_file='UNet_Model_API.png', show_shapes=True)

    print(model.summary())
    return model

def train_model(model,
                dataset_train,
                dataset_valid,
                args
                ):

    '''
    Train the model and save the model
    args:
        model: The compiled model
        dataset_train: The training dataset
        dataset_valid: The validation dataset
        dataset_test: The test dataset
        args: Arguments
    --------------------------------------------------------------------------------
    Returns:
        history: The history of the model
    '''
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                restore_best_weights=True,
                                                min_delta=args.min_delta
                                                
                                                )

    history = model.fit(dataset_train, 
                        epochs=args.epochs, 
                        batch_size=args.batch_size, 
                        validation_data=dataset_valid,
                        verbose=args.verbose, 
                        # steps_per_epoch=args.steps_per_epoch,
                        callbacks=[early_stopping_cb])
    
    # Save model
    model.save("results_sequential/model_%d"%(args.fold))
    return history
    
def save_predictions(dataset_test, model, history, args):
    '''
    Predict the test dataset
    args:
        dataset_test: The test dataset
        model: The compiled model
    --------------------------------------------------------------------------------
    Returns:
        None
    '''
    ins_test =[]
    outs_test =[]
    preds_test =[]
    labels_test =[]

    for i in dataset_test:
        ins_test.append(i[0].numpy())
        outs_test.append(i[1].numpy())
        preds = model.predict(i[0].numpy())
        preds_test.append(preds)
        labels_test.append(np.argmax(preds,axis=3))

    ins_test,outs_test,preds_test,labels_test =   np.concatenate(ins_test, axis=0), np.concatenate(outs_test, axis=0),np.concatenate(preds_test, axis=0), np.concatenate(labels_test, axis=0)
    print(labels_test.shape)    
    results = {}
    results['history'] = history.history
    results['preds_test'] = preds_test
    results['labels_test'] = labels_test
    results['outs_test'] = outs_test
    results['ins_test'] = ins_test

    pickle.dump(results, open("results_sequential/results_fold_%d.pkl"%(args.fold), "wb"))

    
    
    
