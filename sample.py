import numpy as np
from keras.preprocessing import sequence

from parsers.Parse import EnglishPosParser
from vect.Vectorize import Vectorizer

parser = EnglishPosParser()
documents = parser.read_file("eng.train")
vectorizer = Vectorizer("glove.6B.50d.w2v.txt")
features, shapes = vectorizer.encode_features(documents)
labels = vectorizer.encode_annotations(documents)
#print(documents[0].text)
print('Loaded {} data samples'.format(len(features)))


from keras.utils import np_utils

print('Split training/validation')
max_length = 60
# --------------- Features ----------------
x_train, x_validation= [],[]

# For all feature types
# 1. Split features to training and testing set
# 2. Padd sequences
x_= sequence.pad_sequences(features, maxlen=max_length)
x_train, x_validation = np.split(x_, 2)

# --------------- Labels -------------------
y_train, y_validation = [], []

# 1. Convert to one-hot vectors
labels = [np_utils.to_categorical(y_group, num_classes=len(vectorizer.pos_to_index)) for y_group in labels]

# 2. Split labels to training and test set
# 3. (only for sequence tagging) Pad sequences
labels = sequence.pad_sequences(labels, maxlen=max_length)
y_train, y_validation = np.split(labels, 2)


print('Train...')
trained_model_name = 'ner_weights.h5'

# Callback that stops training based on the loss fuction
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Callback that saves the best model across epochs
saveBestModel = ModelCheckpoint(trained_model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.fit(x_train, y_train,
          validation_data=(x_validation, y_validation),
          batch_size=32,  epochs=10, callbacks=[saveBestModel, early_stopping])

# Load the best weights in the model
model.load_weights(trained_model_name)

# Save the complete model
model.save('rnn.h5')