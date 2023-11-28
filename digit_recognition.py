#svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from load_data import read_data
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV

def get_lstm_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(None,3)))
    model.add(tf.keras.layers.LSTM(32, return_sequences=True))
    model.add(tf.keras.layers.LSTM(32))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_conv_model(samples, features):
    """ take in a padded sequence of 3d coordinates, and output a class label
    """
    inputs = tf.keras.Input(shape=(samples,features))
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', kernel_regularizer = "l2")(inputs)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(3)(x)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(128, 3, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def sequences_to_2d(X_train, X_test):
    # Project each sequence to 2d using PCA
    pca = PCA(n_components=2)
    X_train_2d = []
    X_test_2d = []
    for i in range(len(X_train)):
        pca.fit(X_train[i])
        trans = pca.transform(X_train[i])
        #print(trans.shape)
        X_train_2d.append(trans)
    for i in range(len(X_test)):
        pca.fit(X_test[i])
        trans = pca.transform(X_test[i])
        X_test_2d.append(trans)

    pad_amount = max([d.shape[0] for d in X_train_2d])
    X_train = np.zeros((len(X_train_2d),pad_amount,2))
    X_test = np.zeros((len(X_test_2d),pad_amount,2))

    for i in range(len(X_train_2d)):
        X_train[i,:X_train_2d[i].shape[0],:] = X_train_2d[i]
    for i in range(len(X_test_2d)):
        X_test[i,:X_test_2d[i].shape[0],:] = X_test_2d[i]
    return X_train, X_test

if __name__ == "__main__":
    data, labels = read_data()
    data_seq_lens = [len(d) for d in data]
    print(set(data_seq_lens))
    print(labels.shape)
    # Split the data to train and test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

    converted_to_2d = False
    #X_train, X_test = sequences_to_2d(X_train, X_test)

    # Pad X_train and X_test
    pad_amount = max(data_seq_lens)
    # Add zeros (0,0,0) to the end of each sequence, to pad it to the same length
    if converted_to_2d:
        X_train = [np.vstack((d,np.zeros((pad_amount-d.shape[0],2)))) for d in X_train]
        X_test = [np.vstack((d,np.zeros((pad_amount-d.shape[0],2)))) for d in X_test]
    else:
        X_train = [np.vstack((d,np.zeros((pad_amount-d.shape[0],3)))) for d in X_train]
        X_test = [np.vstack((d,np.zeros((pad_amount-d.shape[0],3)))) for d in X_test]
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    print(f"Xtrain shape: {X_train.shape}")
    print(f"Xtest shape: {X_test.shape}")
    
    # Reshape to (samples, features*seq_len)
    X_train = X_train.reshape(X_train.shape[0],-1)
    X_test = X_test.reshape(X_test.shape[0],-1)
    print(f"Xtrain shape: {X_train.shape}")
    print(f"Xtest shape: {X_test.shape}")

    # One hot encode the labels
    #y_train = tf.keras.utils.to_categorical(y_train)
    #y_test = tf.keras.utils.to_categorical(y_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(f"ytrain shape: {y_train.shape}")
    print(f"ytest shape: {y_test.shape}")

    # Normalize the data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    random_forest_param_grid = {
        'n_estimators': [50, 100, 200, 300, 400, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8, None],
        'criterion' :['gini', 'entropy'],
        'bootstrap': [True, False],
        'min_samples_leaf': [1, 2, 4],
    }

    svc_search_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ["rbf"],
        #'degree': [2,3,4,5,6,7]
    }

    knn_search_grid = {
        'n_neighbors': [3,5,11,19],
        'weights': ['uniform','distance'],
        'metric': ['euclidean','manhattan']
    }

    mlp_search_grid = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }



    X_train_trans = X_train
    X_test_trans = X_test

    # Grid search
    model = SVC()
    model = RandomizedSearchCV(model, svc_search_grid, cv=5,scoring='accuracy',verbose=1,n_jobs=-1,n_iter=100)
    #model = get_conv_model(pad_amount,3 if not converted_to_2d else 2)
    # Fit on test data
    model = model.fit(X_train_trans, y_train)
    #early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    #model.fit(X_train_trans,y_train,epochs=200,verbose=1,batch_size=64, validation_data=(X_test,y_test),callbacks=[early_stop])
    #model.evaluate()
    print(model.best_params_)
    conf_mat = confusion_matrix(y_test, model.predict(X_test_trans))
    print(conf_mat)
    print(classification_report(y_test, model.predict(X_test_trans)))
    print(f"Accuracy: ", accuracy_score(y_test,model.predict(X_test)))

    
