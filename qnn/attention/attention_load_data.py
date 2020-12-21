import tensorflow as tf
from sklearn.model_selection import train_test_split

from qnn.attention.attention_word_index_mapping import LanguageIndex, max_length


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
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(sent_pairs, len(lines))

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

# Create batch generator to be used by modle to load data in batches
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
