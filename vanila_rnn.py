import numpy as np
#the input sequence 
x_sequence = np.array([0.5 , 0.2 , -0.1])

#the RNN parameters 
Wx = 0.8
Wh = 0.5 
bh = 0.1 
#define target prediction 
y = 0.4
#define output weight 
Wy = 0.7
by = 0.0
#learning rate
lr = 0.05
epochs = 1000

def forward_pass(x_sequence):
    h_previous = 0.0
    h_states = []
    h_previous_states = []
    for x in x_sequence:
        #save the previous hidden state
        h_previous_states.append(h_previous)
        #combine the current input and the previous hidden state
        z = Wx * x + Wh * h_previous + bh
        #apply the activation function
        h = np.tanh(z)
        #store the hidden state
        h_previous = h
    y_pred = Wy * h_states[-1] + by
    #calculate the loss (mean squared error)
    loss = 0.5 * (y_pred - y) ** 2
    return h_states, h_previous_states, y_pred, loss

