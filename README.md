# Pytorch implementation of Fourier transform of librosa library. 

This codebase implemented discrete Fourier Transform (DFT), inverse DFT as neural network layers in pytorch and can be calculated on GPU. Users can extract log mel spectrogram on GPU. The results are the same as obtained using librosa. The code is developed using pytorch 1.0. 

# Install
```
$ pip install torchlibrosa
```

(Try the following command if the above installa command is not successful)
```
$ pip3 install -i https://test.pypi.org/simple/ torchlibrosa
```


# Examples
```
import torch
import torchlibrosa as tl

# Data
x = torch.zeros(1, 32000)	# (batch_size, samples_num)

# Spectrogram
spectrogram_extractor = tl.stft.Spectrogram(n_fft=1024, hop_length=250)
sp = spectrogram_extractor.forward(x)	# (batch_size, 1, time_steps, freq_bins)

# Log mel spectrogram
logmel_extractor = tl.stft.LogmelFilterBank(sr=32000, n_fft=1024, n_mels=64)
logmel = logmel_extractor.forward(sp)	# (batch_size, 1, time_steps, freq_bins)

# STFT
stft_extractor = tl.stft.STFT(n_fft=1024, hop_length=250)
(real, imag) = stft_extractor.forward(x)
"""real: (batch_size, 1, time_steps, freq_bins), imag: (batch_size, 1, time_steps, freq_bins)"""

# ISTFT
istft_extractor = tl.stft.ISTFT(n_fft=1024, hop_length=250)
y = istft_extractor.forward(real, imag, x.shape[-1])	# (batch_size, samples_num)
```


# Summary
This code provides Fourier transform related functions which can be calculated on GPU using pytorch. 
