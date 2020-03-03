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
