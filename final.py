import sys
import pyaudio
import wave

import scipy
import scipy.io.wavfile

import matplotlib.pyplot as plt

N = 8192
H = N / 4

input_filename = 'piano2.wav'
output_filename = 'output.wav'
rate, data = scipy.io.wavfile.read(input_filename)
left = data[:, 0]
right = data[:, 1]

def main(scaling=1):
  """Scales the pitch of input_filename, while keeping playback time constant."""
  rate, data = scipy.io.wavfile.read(input_filename)
  left = data[:, 0]
  right = data[:, 1]

  left, right = map(lambda data: timescale(data, 1. / scaling), (left, right))
  left, right = map(lambda data: playback_scale(data, scaling), (left, right))
  output = scipy.hstack((left.reshape((len(left), 1)), right.reshape((len(right), 1))))
  scipy.io.wavfile.write(output_filename, rate, output)

  play_audio(output_filename)


def play_audio(filename=output_filename, chunk=1024):
  """Plays the .wav audio file with the given filename."""
  f = wave.open(filename)
  p = pyaudio.PyAudio()

  stream = p.open(format = 
                  p.get_format_from_width(f.getsampwidth()),
                  channels = f.getnchannels(),
                  rate = f.getframerate(),
                  output = True)

  data = f.readframes(chunk)

  while data != '':
    stream.write(data)
    data = f.readframes(chunk)

  stream.close()
  p.terminate()
  f.close()

def display(data):
  plt.plot(data)
  plt.show()

def timescale(data, scaling=1):
  """Scales the playback_duration of input_filename, while keeping pitch constant."""
  length = len(data)

  phi = scipy.zeros(N)
  out = scipy.zeros(N, dtype=complex)
  sigout = scipy.zeros(length / scaling + N)

  amplitude = max(data)
  window = scipy.hanning(N)

  for index in scipy.arange(0, length - (N + H), H * scaling):
    spec1 = scipy.fft(window * data[index : index + N])
    spec2 = scipy.fft(window * data[index + H : index + N + H])

    phi += scipy.angle(spec2 / spec1)
    phi %= 2 * scipy.pi

    out.real, out.imag = scipy.cos(phi), scipy.sin(phi)

    out_index = int(index / scaling)
    sigout[out_index : out_index + N] += (window * scipy.ifft(scipy.absolute(spec2) * out)).real

  sigout *= amplitude / max(sigout)
  return scipy.array(sigout, dtype='int16')

def playback_scale(data, scaling=1):
  """Scales the playback_duration of input_filename, while also scaling pitch."""
  indices = scipy.around(scipy.arange(0, len(data), scaling))
  indices = indices[indices < len(data)]
  return data[indices.astype(int)]


if __name__ == '__main__':
  main(2)
