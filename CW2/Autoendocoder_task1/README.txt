Linear model:
    model.add(Dense(2, name='representation', input_shape=(784, ))
    model.add(Dense(784))
    Test MSE obtained is 0.0557
    clustering accuracy: 0.406
              
Non linear model 1:
    model.add(Dense(128, activation = 'relu', input_shape=(784, )))
    model.add(Dense(2, name='representtion'))
    model.add(Dense(784, activation = 'sigmoid'))
    Test MSE obtained is 0.0545
              
Non linear model 2:
    model.add(Dense(128, activation='relu', input_shape=(784, )))
    model.add(Dense(20, name='representation'))
    model.add(Dense(784, activation = 'sigmoid'))
    Test MSE obtained is 0.0145

Non linear model 3:
    model.add(Dense(128, activation='relu', input_shape(784, )))
    model.add(Dense(50, name='representation'))
    model.add(Dense(784, activation = 'sigmoid'))
    Test MSE obtained is 0.0055
    batch size: 32
    Test MSE obtained is 0.0049
    clustering accuracy: 0.584

Non linear model 4:
    model.add(Dense(128, activation='relu', input_shape=(784, )))
    model.add(Dense(128, activation='relu', input_shape=(784, )))
    model.add(Dense(50, name='representation'))
    model.add(Dense(784, activation = 'sigmoid'))
    Test MSE obtained is 0.064

Non linear model 5
    model.add(Dense(128, activation='tanh', input_shape=(784, )))
    model.add(Dense(50, name='representation'))
    model.add(Dense(784, activation = 'sigmoid'))
    Test MSE obtained is 0.063

Non linear model 6
    model.add(Dense(128, activation='relu', input_shape=(784, )))
    model.add(Dense(50, name='representation'))
    model.add(Dense(784))
    Test MSE obtained is 0.0116
    but the picture becomes gray! Funny... Good point to discuss I think

Non linear model 7
    model.add(Dense(1028, activation='relu', input_shape=(784, )))
    model.add(Dense(128, activation='relu', input_shape= (784, )))
    model.add(Dense(50, name='representation'))
    model.add(Dense(784, activation='sigmoid'))
    batch size: 64
              Test mean squared error: 0.0038
    batch size: 128
              Test mean squared error: 0.0044
    clustering accuracy: 0.619
    
    
    
