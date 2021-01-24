# Pytorch implementation of librosa

This codebase provides PyTorch implementation of some librosa functions. The functions can run on GPU. For example, users can extract log mel spectrogram on GPU. The numerical difference between this codebase and librosa is less than 1e-6.

# Install
```bash
$ pip install torchlibrosa
```

# Examples

Here are examples of extracting spectrogram, log mel spectrogram, STFT and ISTFT using torchlibrosa.

```python
import torch
import torchlibrosa as tl

# Data
x = torch.zeros(1, 22050)	# (batch_size, samples_num)

# Spectrogram
spectrogram_extractor = tl.stft.Spectrogram(n_fft=2048, hop_length=512)
sp = spectrogram_extractor.forward(x)	# (batch_size, 1, time_steps, freq_bins)

# Log mel spectrogram
logmel_extractor = tl.stft.LogmelFilterBank(sr=22050, n_fft=2048, n_mels=128)
logmel = logmel_extractor.forward(sp)	# (batch_size, 1, time_steps, freq_bins)

# STFT
stft_extractor = tl.stft.STFT(n_fft=2048, hop_length=512)
(real, imag) = stft_extractor.forward(x)
# real: (batch_size, 1, time_steps, freq_bins), imag: (batch_size, 1, time_steps, freq_bins) #

# ISTFT
istft_extractor = tl.stft.ISTFT(n_fft=2048, hop_length=512)
y = istft_extractor.forward(real, imag, x.shape[-1])	# (batch_size, samples_num)
```

# More examples

```python
python3 torchlibrosa/stft.py
```

# Compability to librosa functions

If one you previously used for training cpu-extracted features from librosa, but want to add GPU acceleration during i.e., evaluation, then note that the following code will provide identical features to standard mel spectrograms:

```python
## Librosa implementation
import torch
import torchlibrosa as tl

sample_rate = 22050
win_length = 2048
hop_length = 512
n_mels = 128

raw_audio = torch.empty(sample_rate).uniform_(-1, 1) #Float32 input with normalized scale (-1, 1)

#Torchlibrosa feature extractor similar to librosa.feature.melspectrogram()
feature_extractor = torch.nn.Sequential(
    tl.stft.Spectrogram(
        hop_length=hop_length,
        win_length=win_length,
    ), tl.stft.LogmelFilterBank(
        sr=sample_rate,
        n_mels=n_mels,
        is_log=False, #Default is true
    ))
feature = feature_extractor(raw_audio.unsqueeze(0)) # Shape is (Batch, 1, T, N_Mels)
```

# Cite
[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "PANNs: Large-scale pretrained audio neural networks for audio pattern recognition." IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.