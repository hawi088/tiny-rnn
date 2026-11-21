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
Wy = 1.0
by = 0.2
# initial hidden state 
h0 = 0 
#store the hidden states
h_states = []
#priocess thehidden state at each time step 
for x in x_sequence:
    #combine the current input and the previous hidden state
    z = Wx * x + Wh * h0 + bh
    #apply the activation function
    h = np.tanh(z)
    #store the hidden state
    h_states.append(h)
print(h_states)
#calculate the final output
y_pred = Wy * h_states[-1] + by
print("Predicted output:", y_pred)
#calculate the loss (mean squared error)
loss = 0.5 *(y_pred - y) ** 2
print("Loss:", loss)
