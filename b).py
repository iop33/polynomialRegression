%matplotlib inline
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
# Pomocna funkcija koja od niza trening primera pravi feature matricu (m X n).
def create_feature_matrix(x):
  nb_features=3
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)

# Izbegavamo scientific notaciju i zaokruzujemo na 5 decimala.
# np.set_printoptions(suppress=True, precision=5)

# Učitavanje i obrada podataka.
filename = 'funky.csv'
all_data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(0, 1), dtype='float32')
data = dict()
data['x'] = all_data[:, 0]
data['y'] = all_data[:, 1]

# Nasumično mešanje.
nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

# Normalizacija (obratiti pažnju na axis=0).
data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

# Kreiranje feature matrice.
# Ovom promenljivom kontrolisemo broj feature-a tj. stepen polinoma. Varirati!
# nb_features = 1, avg loss u 1000toj epohi = 0.42
# nb_features = 9, avg loss u 1000toj epohi = 0.33
# Overfitting, regularizacija...
nb_features = 3
print('Originalne vrednosti (prve 3):')
print(data['x'][:3])
print('Feature matrica (prva 3 reda):')
data['x'] = create_feature_matrix(data['x'])
print(data['x'][:3, :])




# Model i parametri.
w = tf.Variable(tf.zeros(nb_features))
b = tf.Variable(0.0)

learning_rate = 0.001
nb_epochs = 100

def pred(x, w, b):
    w_col = tf.reshape(w, (nb_features, 1))
    hyp = tf.add(tf.matmul(x, w_col), b)
    return hyp

# Funkcija troška i optimizacija.
def loss(parametar,x, y, w, b, reg=None):
    prediction = pred(x, w, b)

    y_col = tf.reshape(y, (-1, 1))
    mse = tf.reduce_mean(tf.square(prediction - y_col))

    # Regularizacija
    lmbd = parametar
    
    if reg == 'l1':
        l1_reg = lmbd * tf.reduce_mean(tf.abs(w))
        loss = tf.add(mse, l1_reg)
    elif reg == 'l2':
        l2_reg = lmbd * tf.reduce_mean(tf.square(w))
        loss = tf.add(mse, l2_reg)
    else:
        loss = mse
    
    return loss

# Računanje gradijenta
def calc_grad(x, y, w, b,parametar):
    with tf.GradientTape() as tape:
        loss_val = loss(parametar,x, y, w, b, reg='l2')
    
    w_grad, b_grad = tape.gradient(loss_val, [w, b])

    return w_grad, b_grad, loss_val

# Prelazimo na AdamOptimizer jer se prost GradientDescent lose snalazi sa 
# slozenijim funkcijama.
adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Trening korak
def train_step(x, y, w, b,parametar):
    w_grad, b_grad, loss = calc_grad(x, y, w, b,parametar)

    adam.apply_gradients(zip([w_grad, b_grad], [w, b]))

    return loss
plt.scatter(data['x'][:, 0], data['y'])
plt.xlabel('Inputs')
plt.ylabel('Rating')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
wmin=w
bmin=b
lossmin=100000
wrestart=w
brestart=b
variranje = [0, 0.001, 0.01, 0.1, 1, 10, 100]
losses=[]

for parametar in variranje:
        w=wrestart
        b=brestart
          # Trening.
        for epoch in range(nb_epochs):
    
            # Stochastic Gradient Descent.
            epoch_loss = 0
            for sample in range(nb_samples):
                x = data['x'][sample].reshape((1, nb_features))
                y = data['y'][sample]

                curr_loss = train_step(x, y, w, b,parametar)
                epoch_loss += curr_loss
        
            # U svakoj stotoj epohi ispisujemo prosečan loss.
            epoch_loss /= nb_samples

            if (epoch + 1) % 100 == 0:
                print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')
            if(epoch + 1) == 100:
              losses.append(epoch_loss)
              if(lossmin<epoch_loss):
                lossmin=epoch_loss
                wmin=w
                bmin=b

        # Ispisujemo i plotujemo finalnu vrednost parametara.
        print(f'w = {w.numpy()}, bias = {b.numpy()}')
        xs = create_feature_matrix(np.linspace(-2, 4, 100, dtype='float32'))
        hyp_val = pred(xs, w, b)
        plt.plot(xs[:, 0].tolist(), hyp_val.numpy().tolist(), color='g')

plt.show()
plt.figure()
plt.plot(variranje, losses, color='g')
plt.xlabel('Degree')
plt.ylabel('Loss')
plt.show()