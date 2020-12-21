import tensorflow as tf


def gru(units):
	# If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
	# the code automatically does that.
	if tf.test.is_gpu_available():
		return tf.keras.layers.CuDNNGRU(units,
		                                return_sequences=True,
		                                return_state=True,
		                                recurrent_initializer='glorot_uniform')
	else:
		return tf.keras.layers.GRU(units,
		                           return_sequences=True,
		                           return_state=True,
		                           recurrent_activation='sigmoid',
		                           recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
		super(Encoder, self).__init__()
		self.batch_sz = batch_sz
		self.enc_units = enc_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = gru(self.enc_units)

	def call(self, x, hidden):
		x = self.embedding(x)
		output, state = self.gru(x, initial_state=hidden)
		return output, state

	def initialize_hidden_state(self):
		return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
		super(Decoder, self).__init__()
		self.batch_sz = batch_sz
		self.dec_units = dec_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = gru(self.dec_units)
		self.fc = tf.keras.layers.Dense(vocab_size)

		# used for attention
		self.W1 = tf.keras.layers.Dense(self.dec_units)
		self.W2 = tf.keras.layers.Dense(self.dec_units)
		self.V = tf.keras.layers.Dense(1)

	def call(self, x, hidden, enc_output):
		# enc_output shape == (batch_size, max_length, hidden_size)

		# hidden shape == (batch_size, hidden size)
		# hidden_with_time_axis shape == (batch_size, 1, hidden size)
		# we are doing this to perform addition to calculate the score
		hidden_with_time_axis = tf.expand_dims(hidden, 1)

		# score shape == (batch_size, max_length, 1)
		# we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
		# this is the step 1 described in the blog to compute scores s1, s2, ...
		score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

		# attention_weights shape == (batch_size, max_length, 1)
		# this is the step 2 described in the blog to compute attention weights e1, e2, ...
		attention_weights = tf.nn.softmax(score, axis=1)

		# context_vector shape after sum == (batch_size, hidden_size)
		# this is the step 3 described in the blog to compute the context_vector = e1*h1 + e2*h2 + ...
		context_vector = attention_weights * enc_output
		context_vector = tf.reduce_sum(context_vector, axis=1)

		# x shape after passing through embedding == (batch_size, 1, embedding_dim)
		x = self.embedding(x)

		# x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
		# this is the step 4 described in the blog to concatenate the context vector with the output of the previous time step
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

		# passing the concatenated vector to the GRU
		output, state = self.gru(x)

		# output shape == (batch_size * 1, hidden_size)
		output = tf.reshape(output, (-1, output.shape[2]))

		# output shape == (batch_size * 1, vocab)
		# this is the step 5 in the blog, to compute the next output word in the sequence
		x = self.fc(output)

		# return current output, current state and the attention weights
		return x, state, attention_weights

	def initialize_hidden_state(self):
		return tf.zeros((self.batch_sz, self.dec_units))


# Create objects of Class Encoder and Class Decoder
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()


def loss_function(real, pred):
	mask = 1 - np.equal(real, 0)
	loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
	return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
