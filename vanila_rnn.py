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

#trainingloop 
for epoch in range(epochs):
    #forward pass
    h_states, h_previous_states, y_pred, loss = forward_pass(x_sequence)
    #backward pass
    dL_dy = y_pred - y
    #Output layer gradients
    dL_dWy = dL_dy * h_states[-1]
    dL_dby = dL_dy
    #Gradient entering the final hidden state
    dh = dL_dy * Wy
    #initialize RNN parameter gradients
    dWx = 0.0
    dWh = 0.0
    dbh = 0.0
    #Backward pass through time
    for t in reversed(range(len(x_sequence))):
        #Gradient w.r.t. the tanh activation
        dh *= (1 - h_states[t] ** 2)
        #Gradient w.r.t. the bias
        dbh += dh
        #Gradient w.r.t. the hidden-to-hidden weight
        dWh += dh * h_previous_states[t]
        #Gradient w.r.t. the input-to-hidden weight
        dWx += dh * x_sequence[t]
        #Gradient w.r.t. the previous hidden state
        dh = dh * Wh

    #Update RNN parameters
    Wx -= lr * dWx
    Wh -= lr * dWh
    bh -= lr * dbh
    Wy -= lr * dL_dWy
    by -= lr * dL_dby

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, y_pred: {y_pred:.4f}")