"""
Created on Tuesday 20 Février, 2022
@author: N.X.
"""


import numpy as np

class Linear:
    def __init__(self, number_of_entries, number_of_neurons, activation_function):
        self.weights = np.random.randn(number_of_neurons, number_of_entries) * np.sqrt(1/number_of_neurons)
        self.biais = np.zeros((number_of_neurons, 1))
        self.activation_function = activation_function

    def act_fun(self, Z):
        if self.activation_function == "sigmoid" :
            return 1.0 / (1.0 + np.exp(-Z))
        elif self.activation_function == "relu":
            return np.maximum(0, Z)
        elif self.activation_function == "arctan":
            return np.arctan(Z)/np.pi + 0.5
        elif self.activation_function == "tanh":
            return np.tanh(Z)
        return Z

    def deriv_act_fun(self, Z):
        if self.activation_function == "sigmoid" :
            d = 1.0 / (1.0 + np.exp(-Z))
            return d * (1 - d)
        elif self.activation_function == "relu":
            Z[Z > 0] = 1
            Z[Z <= 0] = 0
            return Z
        elif self.activation_function == "arctan":
            return 1/(np.pi*(1 + Z ** 2))
        elif self.activation_function == "tanh":
            return 1 - np.tanh(Z) ** 2
        return 1

    def forward(self, x):
        self.layer_before_activation = []
        self.layer_after_activation = []
        x = self.weights.dot(x) + self.biais
        self.layer_before_activation.append(x)
        x = self.act_fun(x)
        self.layer_after_activation.append(x)
        return x

    def backward(self, previous_layer, delta_l_1, eta):
        delta_l = np.dot(self.weights.T, delta_l_1)* previous_layer.deriv_act_fun(previous_layer.layer_before_activation[0])
        grad_weights = delta_l_1 * previous_layer.layer_after_activation[0].T
        grad_biais = delta_l_1

        self.weights -= eta * grad_weights
        self.biais -= eta * grad_biais

        return delta_l

    def backward_first_layer(self, x, err, eta):
        grad_weights = err * x.T
        grad_biais = err

        delta_1 = (self.weights.T).dot(err)

        self.weights -= eta * grad_weights
        self.biais -= eta * grad_biais

        return delta_1

class LSTM :
    def __init__(self, x_length, h_length):
        self.x_length = x_length
        self.h_length = h_length

        k = np.sqrt(1/self.h_length)

        #timesteps
        self.time_steps = []

        #gradients
        self.gradients_w_i_moyen = np.zeros((self.h_length, self.x_length))
        self.gradients_w_f_moyen = np.zeros((self.h_length, self.x_length))
        self.gradients_w_o_moyen = np.zeros((self.h_length, self.x_length))
        self.gradients_w_c_moyen = np.zeros((self.h_length, self.x_length))

        self.gradients_u_i_moyen = np.zeros((self.h_length, self.h_length))
        self.gradients_u_f_moyen = np.zeros((self.h_length, self.h_length))
        self.gradients_u_o_moyen = np.zeros((self.h_length, self.h_length))
        self.gradients_u_c_moyen = np.zeros((self.h_length, self.h_length))


        self.gradients_b_i_moyen = np.zeros((self.h_length, 1))
        self.gradients_b_f_moyen = np.zeros((self.h_length, 1))
        self.gradients_b_o_moyen = np.zeros((self.h_length, 1))
        self.gradients_b_c_moyen = np.zeros((self.h_length, 1))

        #input gate
        self.Wi = np.random.randn(self.h_length, self.x_length) * k
        self.Ui = np.random.randn(self.h_length, self.h_length) * k
        self.bi = np.random.randn(self.h_length, 1) * k


        #forget gate
        self.Wf = np.random.randn(self.h_length, self.x_length) * k
        self.Uf = np.random.randn(self.h_length, self.h_length) * k
        self.bf = np.random.randn(self.h_length, 1) * k

        #out gate
        self.Wo = np.random.randn(self.h_length, self.x_length) * k
        self.Uo = np.random.randn(self.h_length, self.h_length) * k
        self.bo = np.random.randn(self.h_length, 1) * k

        #cell memory
        self.Wc = np.random.randn(self.h_length, self.x_length) * k
        self.Uc = np.random.randn(self.h_length, self.h_length) * k
        self.bc = np.random.randn(self.h_length, 1) * k


    def forward(self, x_liste):

        h_prec, c_prec = np.zeros((self.h_length, 1)), np.zeros((self.h_length, 1))

        for x in x_liste:

            z_i_t = self.Wi.dot(x) + self.Ui.dot(h_prec) + self.bi
            z_f_t = self.Wf.dot(x) + self.Uf.dot(h_prec) + self.bf
            z_o_t = self.Wo.dot(x) + self.Uo.dot(h_prec) + self.bo
            z_c_t = self.Wc.dot(x) + self.Uc.dot(h_prec) + self.bc

            i = sigmoid(z_i_t)
            f = sigmoid(z_f_t)
            o = sigmoid(z_o_t)
            _c = tanh(z_c_t)
            c = f * c_prec + i * _c
            h = o * tanh(c)


            self.time_steps.append((i, f, o, _c, c, x, h_prec, c_prec))

            (h_prec, c_prec) = (h, c)

        return (h_prec, c_prec)


    def backward_propagation(self, err,eta):
        T = len(self.time_steps)

        err_h = err

        t = T-1

        while t > T - 10 and t >= 0:
            (i, f, o, _c, c, x, h_prec, c_prec) = self.time_steps[t]


            d_c = err_h * o * (1.0 - tanh(c) ** 2)

            delta_f = d_c * c_prec * f * (1 - f)
            delta_i = d_c * _c * i * (1 - i)
            delta_o = err_h * tanh(c) * o * (1 - o)
            delta_c = d_c * i * (1.0 - _c ** 2)

            grad_w_f = delta_f.dot(x.T)
            grad_u_f = delta_f.dot(h_prec.T)
            grad_b_f = delta_f

            grad_w_i = delta_i.dot(x.T)
            grad_u_i = delta_i.dot(h_prec.T)
            grad_b_i = delta_i

            grad_w_o = delta_o.dot(x.T)
            grad_u_o = delta_o.dot(h_prec.T)
            grad_b_o = delta_o

            grad_w_c = delta_c.dot(x.T)
            grad_u_c = delta_c.dot(h_prec.T)
            grad_b_c = delta_c

            k = (T - 1 - t)
            a = 1/(T-t)

            self.gradients_w_i_moyen = a * (k * self.gradients_w_i_moyen + grad_w_i)
            self.gradients_w_f_moyen = a * (k * self.gradients_w_f_moyen + grad_w_f)
            self.gradients_w_c_moyen = a * (k * self.gradients_w_c_moyen + grad_w_c)
            self.gradients_w_o_moyen = a * (k * self.gradients_w_o_moyen + grad_w_o)


            self.gradients_u_i_moyen = a * (k * self.gradients_u_i_moyen + grad_u_i)
            self.gradients_u_f_moyen = a * (k * self.gradients_u_f_moyen + grad_u_f)
            self.gradients_u_c_moyen = a * (k * self.gradients_u_c_moyen + grad_u_c)
            self.gradients_u_o_moyen = a * (k * self.gradients_u_o_moyen + grad_u_o)


            self.gradients_b_i_moyen = a * (k * self.gradients_b_i_moyen + grad_b_i)
            self.gradients_b_f_moyen = a * (k * self.gradients_b_f_moyen + grad_b_f)
            self.gradients_b_c_moyen = a * (k * self.gradients_b_c_moyen + grad_b_c)
            self.gradients_b_o_moyen = a * (k * self.gradients_b_o_moyen + grad_b_o)


            err_h = 1/4 * ((self.Uf.T).dot(delta_f) + (self.Ui.T).dot(delta_i) + (self.Uo.T).dot(delta_o) + (self.Uc.T).dot(delta_c))


            t-=1

        self.Wc -= eta * self.gradients_w_c_moyen
        self.Wf -= eta * self.gradients_w_f_moyen
        self.Wi -= eta * self.gradients_w_i_moyen
        self.Wo -= eta * self.gradients_w_o_moyen

        self.Uc -= eta * self.gradients_u_c_moyen
        self.Uf -= eta * self.gradients_u_f_moyen
        self.Ui -= eta * self.gradients_u_i_moyen
        self.Uo -= eta * self.gradients_u_o_moyen


        self.bc -= eta * self.gradients_b_c_moyen
        self.bf -= eta * self.gradients_b_f_moyen
        self.bi -= eta * self.gradients_b_i_moyen
        self.bo -= eta * self.gradients_b_o_moyen

        self._reset_gradients()
        self.time_steps = []




    def _reset_gradients(self):
        self.gradients_w_i_moyen = np.zeros((self.h_length, self.x_length))
        self.gradients_w_f_moyen = np.zeros((self.h_length, self.x_length))
        self.gradients_w_o_moyen = np.zeros((self.h_length, self.x_length))
        self.gradients_w_c_moyen = np.zeros((self.h_length, self.x_length))

        self.gradients_u_i_moyen = np.zeros((self.h_length, self.h_length))
        self.gradients_u_f_moyen = np.zeros((self.h_length, self.h_length))
        self.gradients_u_o_moyen = np.zeros((self.h_length, self.h_length))
        self.gradients_u_c_moyen = np.zeros((self.h_length, self.h_length))


        self.gradients_b_i_moyen = np.zeros((self.h_length, 1))
        self.gradients_b_f_moyen = np.zeros((self.h_length, 1))
        self.gradients_b_o_moyen = np.zeros((self.h_length, 1))
        self.gradients_b_c_moyen = np.zeros((self.h_length, 1))

class LSTM_Model:
    def __init__(self):
        self.lstm = LSTM(300, 10)
        self.dense_layer = Linear(10, 2, "sigmoid")



    def MSE(self, x, y):
        return 1/2 * np.sum(np.square(x - y))

    def deriv_MSE(self, x, y):
        return (x - y)


    def predict(self, x_list):
        _x = self.lstm.forward(x_list)
        h = _x[0]
        _x = self.dense_layer.forward(_x[0])
        return (h, _x)


    def train(self, X, Y, eta):
        n = len(X)
        s = 0
        for i in range(n):
            x_list = X[i]
            (h, y_prediction) = self.predict(x_list)

            s += self.MSE(y_prediction, Y[i])

            delta_l_1 = self.deriv_MSE(y_prediction, Y[i]) * self.dense_layer.deriv_act_fun(self.dense_layer.layer_before_activation[0])
            err = self.dense_layer.backward_first_layer(h, delta_l_1, eta)

            self.lstm.backward_propagation(err, eta)

        print(s/n)
        

"""
Utilisation de classe LSTM_Model
"""
network = LSTM_Model()

def train(network, data_x, data_y, alpha = 0.001):

    for _ in range(10000):
        network.train(data_x, data_y, alpha)


