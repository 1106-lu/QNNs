import string

import numpy as np
import os
import plotly
import plotly.graph_objects as go
import random
import re
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split

from qnn.attention.attention_word_index_mapping import LanguageIndex, max_length

# Set the file path
file_path = 'C:/Users/usuario/Desktop/QIT/QNNs/qnn/attention/data/spa.txt'  # this might be different in your system

# read the file
lines = open(file_path, encoding='UTF-8').read().strip().split('\n')

# perform basic cleaning
exclude = set(string.punctuation)  # Set of all special characters
remove_digits = str.maketrans('', '', string.digits)  # Set of all digits


def preprocess_eng_sentence(sent):
	'''Function to preprocess English sentence'''
	sent = sent.lower()
	sent = re.sub("'", '', sent)
	sent = ''.join(ch for ch in sent if ch not in exclude)
	sent = sent.translate(remove_digits)
	sent = sent.strip()
	sent = re.sub(" +", " ", sent)
	sent = '<start> ' + sent + ' <end>'
	return sent


def preprocess_spa_sentence(sent):
	'''Function to preprocess Spanish sentence'''
	sent = sent.lower()
	sent = re.sub("'", '', sent)
	sent = ''.join(ch for ch in sent if ch not in exclude)
	sent = sent.translate(remove_digits)
	sent = sent.strip()
	sent = re.sub(" +", " ", sent)
	sent = '<start> ' + sent + ' <end>'
	return sent


# Generate pairs of cleaned English and Spanish sentences
sent_pairs = []
i = 0
for line in lines:
	i += 1
	sent_pair = []
	eng, spa, _ = line.split('\t')
	eng = preprocess_eng_sentence(eng)
	sent_pair.append(eng)
	spa = preprocess_spa_sentence(spa)
	sent_pair.append(spa)
	sent_pairs.append(sent_pair)

random.shuffle(sent_pairs)
sent_pairs_input = sent_pairs[1000:2000]


#  if i == 10000:
#  	break
#  else:
#  	pass
# return sent


def load_dataset(pairs, num_examples):
	# pairs => already created cleaned input, output pairs

	# index language using the class defined above
	inp_lang = LanguageIndex(en for en, ma in pairs)
	targ_lang = LanguageIndex(ma for en, ma in pairs)

	# Vectorize the input and target languages

	# English sentences
	input_tensor = [[inp_lang.word2idx[s] for s in en.split(' ')] for en, ma in pairs]

	# Marathi sentences
	target_tensor = [[targ_lang.word2idx[s] for s in ma.split(' ')] for en, ma in pairs]

	# Calculate max_length of input and output tensor
	# Here, we'll set those to the longest sentence in the dataset
	max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

	# Padding the input and output tensor to the maximum length
	input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
	                                                             maxlen=max_length_inp,
	                                                             padding='post')

	target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
	                                                              maxlen=max_length_tar,
	                                                              padding='post')

	return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar


# Create the tensors
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(sent_pairs_input,
                                                                                                 len(lines))

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.1,
                                                                                                random_state=101)

# Set the parameters of the model
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

print(BUFFER_SIZE, BATCH_SIZE, N_BATCH, vocab_inp_size, vocab_tar_size)

# Create batch generator to be used by modle to load data in batches
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

print('Data set prepared')


class LanguageIndex():
	def __init__(self, lang):
		self.lang = lang
		self.word2idx = {}
		self.idx2word = {}
		self.vocab = set()

		self.create_index()

	def create_index(self):
		for phrase in self.lang:
			self.vocab.update(phrase.split(' '))

		self.vocab = sorted(self.vocab)

		self.word2idx['<pad>'] = 0
		for index, word in enumerate(self.vocab):
			self.word2idx[word] = index + 1

		for word, index in self.word2idx.items():
			self.idx2word[index] = word


# Function to calculate maximum length of the sequence
def max_length(tensor):
	return max(len(t) for t in tensor)


def gru(units):
	# If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
	# the code automatically does that.
	if tf.test.is_gpu_available():
		pass
	# return tf.keras.layers.CuDNNGRU(units,
	#                                 return_sequences=True,
	#                                 return_state=True,
	#                                 recurrent_initializer='glorot_uniform')
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

	def __call__(self, x, hidden):
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

	def __call__(self, x, hidden, enc_output):
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

optimizer = tf.keras.optimizers.Adam()


def loss_function(real, pred):
	mask = 1 - np.equal(real, 0)
	loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
	return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

print('Model defined')

EPOCHS = 35
for epoch in range(EPOCHS):
	start = time.time()
	hidden = encoder.initialize_hidden_state()
	total_loss = 0
	for (batch, (inp, targ)) in enumerate(dataset):
		loss = 0
		with tf.GradientTape() as tape:
			enc_output, enc_hidden = encoder(inp, hidden)
			dec_hidden = enc_hidden
			dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)
			# Teacher forcing - feeding the target as the next input
			for t in range(1, targ.shape[1]):
				# passing enc_output to the decoder
				predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
				loss += loss_function(targ[:, t], predictions)
				# using teacher forcing
				dec_input = tf.expand_dims(targ[:, t], 1)
		batch_loss = (loss / int(targ.shape[1]))
		total_loss += batch_loss
		variables = encoder.variables + decoder.variables
		gradients = tape.gradient(loss, variables)
		optimizer.apply_gradients(zip(gradients, variables))
		if batch % 100 == 0:
			print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
			                                             batch,
			                                             batch_loss.numpy()))
	# saving (checkpoint) the model every epoch
	checkpoint.save(file_prefix=checkpoint_prefix)
	print('Epoch {} Loss {:.4f}'.format(epoch + 1,
	                                    total_loss / N_BATCH))
	print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
def evaluate(inputs, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
	attention_plot = np.zeros((max_length_targ, max_length_inp))
	sentence = ''
	for i in inputs[0]:
		if i == 0:
			break
		sentence = sentence + inp_lang.idx2word[i] + ' '
	sentence = sentence[:-1]
	inputs = tf.convert_to_tensor(inputs)
	result = ''
	hidden = [tf.zeros((1, units))]
	enc_out, enc_hidden = encoder(inputs, hidden)
	dec_hidden = enc_hidden
	dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)
	# start decoding
	for t in range(max_length_targ):  # limit the length of the decoded sequence
		predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
		# storing the attention weights to plot later on
		attention_weights = tf.reshape(attention_weights, (-1,))
		attention_plot[t] = attention_weights.numpy()
		predicted_id = tf.argmax(predictions[0]).numpy()
		result += targ_lang.idx2word[predicted_id] + ' '
		# stop decoding if '<end>' is predicted
		if targ_lang.idx2word[predicted_id] == '<end>':
			return result, sentence, attention_plot
		# the predicted ID is fed back into the model
		dec_input = tf.expand_dims([predicted_id], 0)
	return result, sentence, attention_plot
def predict_random_val_sentence():
	actual_sent = ''
	k = np.random.randint(len(input_tensor_val))
	random_input = input_tensor_val[k]
	random_output = target_tensor_val[k]
	random_input = np.expand_dims(random_input, 0)
	result, sentence, attention_plot = evaluate(random_input, encoder, decoder, inp_lang, targ_lang, max_length_inp,
	                                            max_length_targ)
	print('Input: {}'.format(sentence[8:-6]))
	print('Predicted translation: {}'.format(result[:-6]))
	for i in random_output:
		if i == 0:
			break
		actual_sent = actual_sent + targ_lang.idx2word[i] + ' '
	actual_sent = actual_sent[8:-7]
	print('Actual translation: {}'.format(actual_sent))
	attention_plot = attention_plot[:len(result.split(' ')) - 2, 1:len(sentence.split(' ')) - 1]
	sentence, result = sentence.split(' '), result.split(' ')
	sentence = sentence[1:-1]
	result = result[:-2]
	# use plotly to plot the heatmap
	trace = go.Heatmap(z=attention_plot, x=sentence, y=result, colorscale='Reds')
	data = [trace]
	plotly.offline.iplot(data)
# Finally call the function multiple times to visualize random results from the test set
for _ in range(10):
	predict_random_val_sentence()
