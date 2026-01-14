import os
import librosa
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

CAPUCHIN_FILE = os.path.join('Sounds_of_birds','Parsed_Capuchinbird_Clips', 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join('Sounds_of_birds', 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forest-0.wav')

#loading the audio files in
C_F,s= librosa.load(CAPUCHIN_FILE,sr = 16000)
NC_F , _ =librosa.load(NOT_CAPUCHIN_FILE , sr = 16000)
# sampling rate set by the librosa.load method
print(s)
print(len(C_F),len(NC_F))
#Visualising the wave forms
plt.figure(figsize=(14, 5))
plt.plot(C_F)
plt.plot(NC_F)
plt.title('Waveform of the Audio Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# creating the dataset and label positive = 1 & negative = 0
POS = os.path.join('Sounds_of_birds', 'Parsed_Capuchinbird_Clips')
NEG = os.path.join('Sounds_of_birds', 'Parsed_Not_Capuchinbird_Clips')

pos = tf.data.Dataset.list_files(POS + '/*.wav')
neg = tf.data.Dataset.list_files(NEG + '/*.wav')

positives = pos.map(lambda x: (x, 1))
negatives = neg.map(lambda x: (x, 0))

data = positives.concatenate(negatives)
print(data)

t = data.as_numpy_iterator().next()
print(t)

# pre processing function to make MFCC
def preprocess(filepath, label):
    filepath = filepath.numpy().decode("utf-8")
    wave, sr = librosa.load(filepath, sr=16000)
    wave = wave[:48000]

    # padding
    padding_len = 48000 - len(wave)
    if padding_len > 0:
        wave = np.pad(wave, (0, padding_len))

    # MFCC
    mfcc = librosa.feature.mfcc(y=wave, n_mfcc=13, sr=sr)
    mfcc = mfcc.T  # (time, n_mfcc)
    mfcc = mfcc[..., None]  # (time, n_mfcc, 1)

    return mfcc.astype(np.float32), label

#visulalising data , how does an actual call look v/s noise using mfcc plots
# filepath_ , label_ = negatives.shuffle(buffer_size=1000).as_numpy_iterator().next()
# Mfcc, label = preprocess(filepath_, label_)
# mfcc_plot = Mfcc[..., 0]
# print(mfcc_plot.shape)
#
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(mfcc_plot.T, x_axis='time',sr = 16000)
# plt.colorbar(format='%+2.0f')
# plt.show()


def tf_preprocess(x, y):
    mfcc, label = tf.py_function(
        preprocess,
        [x, y],
        [tf.float32, tf.int32]
    )

    mfcc.set_shape([None, 13, 1])
    label.set_shape([])

    return mfcc, label


data = data.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
data = data.cache()
data = data.shuffle(1000)

train = data.take(36)
val   = data.skip(36).take(15)

train = train.batch(16)
val   = val.batch(16)

train = train.prefetch(tf.data.AUTOTUNE)
val   = val.prefetch(tf.data.AUTOTUNE)

sample, label = next(iter(train))
print(sample.shape)

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(94,13,1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

model.fit(train, validation_data=val, epochs=10)
print(model.summary())


X_test, y_test = val.as_numpy_iterator().next()
print(X_test)
yhat = model.predict(X_test)


yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
print(tf.math.reduce_sum(yhat))
print(tf.math.reduce_sum(y_test))

model.export('capuchin_model')

converter = tf.lite.TFLiteConverter.from_saved_model("capuchin_model")
tflite_model = converter.convert()

with open("capuchin_float.tflite", "wb") as f:
    f.write(tflite_model)



def representative_data_gen():
    for mfcc, _ in train.take(50):
        yield [mfcc]


converter = tf.lite.TFLiteConverter.from_saved_model("capuchin_model")

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open("capuchin_int8.tflite", "wb") as f:
    f.write(tflite_quant_model)


mfcc, label = next(iter(val.unbatch().take(1)))


mfcc_batch = tf.expand_dims(mfcc, axis=0)

# Keras prediction (float32)
keras_pred = model.predict(mfcc_batch)
print("Keras output:", keras_pred)
print("Keras class:", int(keras_pred > 0.5))

#predicting using quantized model
interpreter = tf.lite.Interpreter("capuchin_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Quantize input
in_scale, in_zero = input_details[0]['quantization']
mfcc_q = mfcc / in_scale + in_zero
mfcc_q = mfcc_q.numpy().astype(np.int8)
mfcc_q = np.expand_dims(mfcc_q, axis=0)


interpreter.set_tensor(input_details[0]['index'], mfcc_q)
interpreter.invoke()


output_q = interpreter.get_tensor(output_details[0]['index'])

# Dequantize output
out_scale, out_zero = output_details[0]['quantization']
tflite_pred = out_scale * (output_q - out_zero)

print("TFLite output:", tflite_pred)
print("TFLite class:", int(tflite_pred > 0.5))

#see difference between outputs
diff = abs(keras_pred - tflite_pred)
print("Absolute difference:", diff)

