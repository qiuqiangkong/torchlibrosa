import numpy as np
import librosa
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class DFTBase(nn.Module):
    def __init__(self):
        """Base class for DFT and IDFT matrix"""
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W


class DFT(DFTBase):
    def __init__(self, n, norm):
        """Calculate DFT, IDFT, RDFT, IRDFT. 
        Args:
          n: fft window size
          norm: None | 'ortho'
        """
        super(DFT, self).__init__()

        self.W = self.dft_matrix(n)
        self.inv_W = self.idft_matrix(n)

        self.W_real = torch.Tensor(np.real(self.W))
        self.W_imag = torch.Tensor(np.imag(self.W))
        self.inv_W_real = torch.Tensor(np.real(self.inv_W))
        self.inv_W_imag = torch.Tensor(np.imag(self.inv_W))

        self.n = n
        self.norm = norm

    def dft(self, x_real, x_imag):
        """Calculate DFT of signal. 
        Args:
          x_real: (n,), signal real part
          x_imag: (n,), signal imag part
        Returns:
          z_real: (n,), output real part
          z_imag: (n,), output imag part
        """
        z_real = torch.matmul(x_real, self.W_real) - torch.matmul(x_imag, self.W_imag)
        z_imag = torch.matmul(x_imag, self.W_real) + torch.matmul(x_real, self.W_imag)

        if self.norm is None:
            pass
        elif self.norm == 'ortho':
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

        return z_real, z_imag

    def idft(self, x_real, x_imag):
        """Calculate IDFT of signal. 
        Args:
          x_real: (n,), signal real part
          x_imag: (n,), signal imag part
        Returns:
          z_real: (n,), output real part
          z_imag: (n,), output imag part
        """
        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(x_imag, self.inv_W_imag)
        z_imag = torch.matmul(x_imag, self.inv_W_real) + torch.matmul(x_real, self.inv_W_imag)

        if self.norm is None:
            z_real /= self.n
        elif self.norm == 'ortho':
            z_real /= math.sqrt(n)
            z_imag /= math.sqrt(n)

        return z_real, z_imag

    def rdft(self, x_real):
        """Calculate right DFT of signal. 
        Args:
          x_real: (n,), signal real part
          x_imag: (n,), signal imag part
        Returns:
          z_real: (n // 2 + 1,), output real part
          z_imag: (n // 2 + 1,), output imag part
        """
        n_rfft = self.n // 2 + 1
        z_real = torch.matmul(x_real, self.W_real[..., 0 : n_rfft])
        z_imag = torch.matmul(x_real, self.W_imag[..., 0 : n_rfft])

        if self.norm is None:
            pass
        elif self.norm == 'ortho':
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

    def irdft(self, x_real, x_imag):
        """Calculate inverse right DFT of signal. 
        Args:
          x_real: (n // 2 + 1,), signal real part
          x_imag: (n // 2 + 1,), signal imag part
        Returns:
          z_real: (n,), output real part
          z_imag: (n,), output imag part
        """
        n_rfft = self.n // 2 + 1

        flip_x_real = torch.flip(x_real, dims=(-1,))
        x_real = torch.cat((x_real, flip_x_real[..., 1 : n_rfft - 1]), dim=-1)

        flip_x_imag = torch.flip(x_imag, dims=(-1,))
        x_imag = torch.cat((x_imag, -1. * flip_x_imag[..., 1 : n_rfft - 1]), dim=-1)

        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(x_imag, self.inv_W_imag)

        if self.norm is None:
            z_real /= self.n
        elif self.norm == 'ortho':
            z_real /= math.sqrt(n)

        return z_real


class STFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """Implementation of STFT with Conv1d. The function has the same output 
        of librosa.core.stft
        """
        super(STFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if self.win_length is None:
            self.win_length = n_fft

        # Set the default hop, if it's not already specified
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        fft_window = librosa.filters.get_window(window, self.win_length, fftbins=True)

        # Pad the window out to n_fft size
        fft_window = librosa.util.pad_center(fft_window, n_fft)

        # DFT & IDFT matrix
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channels,
            kernel_size=n_fft, stride=self.hop_length, padding=0, dilation=1,
            groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channels,
            kernel_size=n_fft, stride=self.hop_length, padding=0, dilation=1,
            groups=1, bias=False)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, data_length)
        Returns:
          real: (batch_size, n_fft // 2 + 1, time_steps)
          imag: (batch_size, n_fft // 2 + 1, time_steps)
        """

        x = input[:, None, :]   # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag


def magphase(real, imag):
    mag = (real ** 2 + imag ** 2) ** 0.5
    cos = real / torch.clamp(mag, 1e-10, np.inf)
    sin = imag / torch.clamp(mag, 1e-10, np.inf)
    return mag, cos, sin


class ISTFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """Implementation of ISTFT with Conv1d. The function has the same output 
        of librosa.core.istft
        """
        super(ISTFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if self.win_length is None:
            self.win_length = self.n_fft

        # Set the default hop, if it's not already specified
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        # DFT & IDFT matrix
        self.W = self.idft_matrix(n_fft) / n_fft

        self.conv_real = nn.Conv1d(in_channels=n_fft, out_channels=n_fft,
            kernel_size=1, stride=1, padding=0, dilation=1,
            groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=n_fft, out_channels=n_fft,
            kernel_size=1, stride=1, padding=0, dilation=1,
            groups=1, bias=False)

        self.reverse = nn.Conv1d(in_channels=n_fft // 2 + 1,
            out_channels=n_fft // 2 - 1, kernel_size=1, bias=False)

        self.overlap_add = nn.ConvTranspose2d(in_channels=n_fft,
            out_channels=1, kernel_size=(n_fft, 1), stride=(self.hop_length, 1), bias=False)

        self.ifft_window_sum = []

        self.init_weights()

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def init_weights(self):
        ifft_window = librosa.filters.get_window(self.window, self.win_length, fftbins=True)
        """(win_length,)"""

        # Pad the window to n_fft
        ifft_window = librosa.util.pad_center(ifft_window, self.n_fft)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W * ifft_window[None, :]).T)[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W * ifft_window[None, :]).T)[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

        tmp = np.zeros((self.n_fft // 2 - 1, self.n_fft // 2 + 1, 1))
        tmp[:, 1 : -1, 0] = np.array(np.eye(self.n_fft // 2 - 1)[::-1])
        self.reverse.weight.data = torch.Tensor(tmp)
        """(n_fft // 2 - 1, n_fft // 2 + 1, 1)"""

        self.overlap_add.weight.data = torch.Tensor(np.eye(self.n_fft)[:, None, :, None])
        """(n_fft, 1, n_fft, 1)"""

    def get_ifft_window(self, n_frames):
        device = next(self.parameters()).device

        ifft_window_sum = librosa.filters.window_sumsquare(self.window, n_frames,
            win_length=self.win_length, n_fft=self.n_fft, hop_length=self.hop_length)

        ifft_window_sum = np.clip(ifft_window_sum, 1e-8, np.inf)
        ifft_window_sum = torch.Tensor(ifft_window_sum).to(device)
        return ifft_window_sum

    def forward(self, real_stft, imag_stft, length):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)
        Returns:
          real: (batch_size, data_length)
        """
        assert real_stft.ndimension() == 4 and imag_stft.ndimension() == 4
        device = next(self.parameters()).device
        batch_size = real_stft.shape[0]

        real_stft = real_stft[:, 0, :, :].transpose(1, 2)
        imag_stft = imag_stft[:, 0, :, :].transpose(1, 2)
        # (batch_size, n_fft // 2 + 1, time_steps)

        # Full stft, using flip is not supported by ONNX.
        # full_real_stft = torch.cat((real_stft, torch.flip(real_stft[:, 1 : -1, :], dims=[1])), dim=1)
        # full_imag_stft = torch.cat((imag_stft, - torch.flip(imag_stft[:, 1 : -1, :], dims=[1])), dim=1)
        full_real_stft = torch.cat((real_stft, self.reverse(real_stft)), dim=1)
        full_imag_stft = torch.cat((imag_stft, - self.reverse(imag_stft)), dim=1)
        """(1, n_fft, time_steps)"""

        # IDFT
        s_real = self.conv_real(full_real_stft) - self.conv_imag(full_imag_stft)
        s_real = s_real[..., None]  # (1, n_fft, time_steps, 1)
        y = self.overlap_add(s_real)[:, 0, :, 0]    # (1, samples_num)

        # Divide window
        if len(self.ifft_window_sum) != y.shape[1]:
            frames_num = real_stft.shape[2]
            self.ifft_window_sum = self.get_ifft_window(frames_num)

        y = y / self.ifft_window_sum[None, 0 : y.shape[1]]

        # Trim or pad to length
        if length is None:
            if self.center:
                y = y[:, self.n_fft // 2 : -self.n_fft // 2]
        else:
            if self.center:
                start = self.n_fft // 2
            else:
                start = 0

            y = y[:, start : start + length]
            (batch_size, len_y) = y.shape
            if y.shape[-1] < length:
                y = torch.cat((y, torch.zeros(batch_size, length - len_y).to(device)), dim=-1)

        return y


class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
        window='hann', center=True, pad_mode='reflect', power=2.0,
        freeze_parameters=True):
        """Calculate spectrogram using pytorch. The STFT is implemented with 
        Conv1d. The function has the same output of librosa.core.stft
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center,
            pad_mode=pad_mode, freeze_parameters=True)

    def forward(self, input):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)
        Returns:
          spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real ** 2 + imag ** 2

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (self.power / 2.0)

        return spectrogram


class LogmelFilterBank(nn.Module):
    def __init__(self, sr=22050, n_fft=2048, n_mels=64, fmin=0.0, fmax=None, is_log=True,
        ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        """Calculate logmel spectrogram using pytorch. The mel filter bank is 
        the pytorch implementation of as librosa.filters.mel 
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        if fmax == None:
            fmax = sr//2


        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, channels, time_steps)
        
        Output: (batch_size, time_steps, mel_bins)
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output


    def power_to_db(self, input):
        """Power to db, this function is the pytorch implementation of 
        librosa.core.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise librosa.util.exceptions.ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db, max=np.inf)

        return log_spec


class Enframe(nn.Module):
    def __init__(self, frame_length=2048, hop_length=512):
        """Enframe a time sequence. This function is the pytorch implementation 
        of librosa.util.frame
        """
        super(Enframe, self).__init__()

        self.enframe_conv = nn.Conv1d(in_channels=1, out_channels=frame_length,
            kernel_size=frame_length, stride=hop_length,
            padding=0, bias=False)

        self.enframe_conv.weight.data = torch.Tensor(torch.eye(frame_length)[:, None, :])
        self.enframe_conv.weight.requires_grad = False

    def forward(self, input):
        """input: (batch_size, samples)
        
        Output: (batch_size, window_length, frames_num)
        """
        output = self.enframe_conv(input[:, None, :])
        return output


    def power_to_db(self, input):
        """Power to db, this function is the pytorch implementation of 
        librosa.core.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise librosa.util.exceptions.ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(log_spec, min=log_spec.max() - self.top_db, max=np.inf)

        return log_spec


class Scalar(nn.Module):
    def __init__(self, scalar, freeze_parameters):
        super(Scalar, self).__init__()

        self.scalar_mean = Parameter(torch.Tensor(scalar['mean']))
        self.scalar_std = Parameter(torch.Tensor(scalar['std']))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        return (input - self.scalar_mean) / self.scalar_std


def debug(select, device):
    """Compare numpy + librosa and torchlibrosa results. For debug. 

    Args:
      select: 'dft' | 'logmel'
      device: 'cpu' | 'cuda'
    """

    if select == 'dft':
        n = 10
        norm = None     # None | 'ortho'
        np.random.seed(0)

        # Data
        np_data = np.random.uniform(-1, 1, n)
        pt_data = torch.Tensor(np_data)

        # Numpy FFT
        np_fft = np.fft.fft(np_data, norm=norm)
        np_ifft = np.fft.ifft(np_fft, norm=norm)
        np_rfft = np.fft.rfft(np_data, norm=norm)
        np_irfft = np.fft.ifft(np_rfft, norm=norm)

        # Pytorch FFT
        obj = DFT(n, norm)
        pt_dft = obj.dft(pt_data, torch.zeros_like(pt_data))
        pt_idft = obj.idft(pt_dft[0], pt_dft[1])
        pt_rdft = obj.rdft(pt_data)
        pt_irdft = obj.irdft(pt_rdft[0], pt_rdft[1])

        print('Comparing librosa and pytorch implementation of DFT. All numbers '
            'below should be close to 0.')
        print(np.mean((np.abs(np.real(np_fft) - pt_dft[0].cpu().numpy()))))
        print(np.mean((np.abs(np.imag(np_fft) - pt_dft[1].cpu().numpy()))))

        print(np.mean((np.abs(np.real(np_ifft) - pt_idft[0].cpu().numpy()))))
        print(np.mean((np.abs(np.imag(np_ifft) - pt_idft[1].cpu().numpy()))))

        print(np.mean((np.abs(np.real(np_rfft) - pt_rdft[0].cpu().numpy()))))
        print(np.mean((np.abs(np.imag(np_rfft) - pt_rdft[1].cpu().numpy()))))

        print(np.mean(np.abs(np_data - pt_irdft.cpu().numpy())))

    elif select == 'stft':
        device = torch.device(device)
        np.random.seed(0)

        # Spectrogram parameters (the same as librosa.stft)
        sample_rate = 22050
        data_length = sample_rate * 1
        n_fft = 2048
        hop_length = 512
        win_length = 2048
        window = 'hann'
        center = True
        pad_mode = 'reflect'

        # Data
        np_data = np.random.uniform(-1, 1, data_length)
        pt_data = torch.Tensor(np_data).to(device)

        # Numpy stft matrix
        np_stft_matrix = librosa.core.stft(y=np_data, n_fft=n_fft,
            hop_length=hop_length, window=window, center=center).T

        # Pytorch stft matrix
        pt_stft_extractor = STFT(n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        pt_stft_extractor.to(device)

        (pt_stft_real, pt_stft_imag) = pt_stft_extractor.forward(pt_data[None, :])

        print('Comparing librosa and pytorch implementation of DFT. All numbers '
            'below should be close to 0.')
        print(np.mean(np.abs(np.real(np_stft_matrix) - pt_stft_real.data.cpu().numpy()[0, 0])))
        print(np.mean(np.abs(np.imag(np_stft_matrix) - pt_stft_imag.data.cpu().numpy()[0, 0])))

        # Numpy istft
        np_istft_s = librosa.core.istft(stft_matrix=np_stft_matrix.T,
            hop_length=hop_length, window=window, center=center, length=data_length)

        # Pytorch istft
        pt_istft_extractor = ISTFT(n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)
        pt_istft_extractor.to(device)

        # Recover from real and imag part
        pt_istft_s = pt_istft_extractor.forward(pt_stft_real, pt_stft_imag, data_length)[0, :]

        # Recover from magnitude and phase
        (pt_stft_mag, cos, sin) = magphase(pt_stft_real, pt_stft_imag)
        pt_istft_s2 = pt_istft_extractor.forward(pt_stft_mag * cos, pt_stft_mag * sin, data_length)[0, :]

        print(np.mean(np.abs(np_istft_s - pt_istft_s.data.cpu().numpy())))
        print(np.mean(np.abs(np_data - pt_istft_s.data.cpu().numpy())))
        print(np.mean(np.abs(np_data - pt_istft_s2.data.cpu().numpy())))

    elif select == 'logmel':
        dtype = np.complex64
        device = torch.device(device)
        np.random.seed(0)

        # Spectrogram parameters (the same as librosa.stft)
        sample_rate = 22050
        data_length = sample_rate * 1
        n_fft = 2048
        hop_length = 512
        win_length = 2048
        window = 'hann'
        center = True
        pad_mode = 'reflect'

        # Mel parameters (the same as librosa.feature.melspectrogram)
        n_mels = 128
        fmin = 0.
        fmax = sample_rate / 2.0

        # Power to db parameters (the same as default settings of librosa.power_to_db
        ref = 1.0
        amin = 1e-10
        top_db = 80.0

        # Data
        np_data = np.random.uniform(-1, 1, data_length)
        pt_data = torch.Tensor(np_data).to(device)

        print('Comparing librosa and pytorch implementation of logmel '
            'spectrogram. All numbers below should be close to 0.')

        # Numpy librosa
        np_stft_matrix = librosa.core.stft(y=np_data, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, dtype=dtype,
            pad_mode=pad_mode)

        np_pad = np.pad(np_data, int(n_fft // 2), mode=pad_mode)

        np_melW = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax).T

        np_mel_spectrogram = np.dot(np.abs(np_stft_matrix.T) ** 2, np_melW)

        np_logmel_spectrogram = librosa.core.power_to_db(
            np_mel_spectrogram, ref=ref, amin=amin, top_db=top_db)

        # Pytorch
        stft_extractor = STFT(n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft,
            n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
            top_db=top_db, freeze_parameters=True)

        stft_extractor.to(device)
        logmel_extractor.to(device)

        pt_pad = F.pad(pt_data[None, None, :], pad=(n_fft // 2, n_fft // 2), mode=pad_mode)[0, 0]
        print(np.mean(np.abs(np_pad - pt_pad.cpu().numpy())))

        pt_stft_matrix_real = stft_extractor.conv_real(pt_pad[None, None, :])[0]
        pt_stft_matrix_imag = stft_extractor.conv_imag(pt_pad[None, None, :])[0]
        print(np.mean(np.abs(np.real(np_stft_matrix) - pt_stft_matrix_real.data.cpu().numpy())))
        print(np.mean(np.abs(np.imag(np_stft_matrix) - pt_stft_matrix_imag.data.cpu().numpy())))

        # Spectrogram
        spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        spectrogram_extractor.to(device)

        pt_spectrogram = spectrogram_extractor.forward(pt_data[None, :])
        pt_mel_spectrogram = torch.matmul(pt_spectrogram, logmel_extractor.melW)
        print(np.mean(np.abs(np_mel_spectrogram - pt_mel_spectrogram.data.cpu().numpy()[0, 0])))

        # Log mel spectrogram
        pt_logmel_spectrogram = logmel_extractor.forward(pt_spectrogram)
        print(np.mean(np.abs(np_logmel_spectrogram - pt_logmel_spectrogram[0, 0].data.cpu().numpy())))

    elif select == 'enframe':
        device = torch.device(device)
        np.random.seed(0)

        # Spectrogram parameters (the same as librosa.stft)
        sample_rate = 22050
        data_length = sample_rate * 1
        hop_length = 512
        win_length = 2048

        # Data
        np_data = np.random.uniform(-1, 1, data_length)
        pt_data = torch.Tensor(np_data).to(device)

        print('Comparing librosa and pytorch implementation of '
            'librosa.util.frame. All numbers below should be close to 0.')

        # Numpy librosa
        np_frames = librosa.util.frame(np_data, frame_length=win_length,
            hop_length=hop_length)

        # Pytorch
        pt_frame_extractor = Enframe(frame_length=win_length, hop_length=hop_length)
        pt_frame_extractor.to(device)

        pt_frames = pt_frame_extractor(pt_data[None, :])
        print(np.mean(np.abs(np_frames - pt_frames.data.cpu().numpy())))

    elif select == 'default':
        device = torch.device(device)
        np.random.seed(0)

        # Spectrogram parameters (the same as librosa.stft)
        sample_rate = 22050
        data_length = sample_rate * 1
        hop_length = 512
        win_length = 2048

        # Mel parameters (the same as librosa.feature.melspectrogram)
        n_mels = 128

        # Data
        np_data = np.random.uniform(-1, 1, data_length)
        pt_data = torch.Tensor(np_data).to(device)
        feature_extractor = nn.Sequential(
            Spectrogram(
                hop_length=hop_length,
                win_length=win_length,
            ), LogmelFilterBank(
                sr=sample_rate,
                n_mels=n_mels,
                is_log=False, #Default is true
            ))
        feature_extractor.to(device)

        print(
            'Comparing default mel spectrogram from librosa to the pytorch implementation.'
        )

        # Numpy librosa
        np_melspect = librosa.feature.melspectrogram(np_data,
                                                     hop_length=hop_length,
                                                     sr=sample_rate,
                                                     win_length=win_length,
                                                     n_mels=n_mels).T
        #Pytorch
        pt_melspect = feature_extractor(pt_data[None, :]).squeeze()
        passed = np.allclose(pt_melspect.data.to('cpu').numpy(), np_melspect)
        print(f"Passed? {passed}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    device = args.device
    norm = None     # None | 'ortho'
    np.random.seed(0)

    # Spectrogram parameters (the same as librosa.stft)
    sample_rate = 22050
    data_length = sample_rate * 1
    n_fft = 2048
    hop_length = 512
    win_length = 2048
    window = 'hann'
    center = True
    pad_mode = 'reflect'

    # Mel parameters (the same as librosa.feature.melspectrogram)
    n_mels = 128
    fmin = 0.
    fmax = sample_rate / 2.0

    # Power to db parameters (the same as default settings of librosa.power_to_db
    ref = 1.0
    amin = 1e-10
    top_db = 80.0

    # Data
    np_data = np.random.uniform(-1, 1, data_length)
    pt_data = torch.Tensor(np_data).to(device)

    # Pytorch
    spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=window, center=center, pad_mode=pad_mode,
        freeze_parameters=True)

    logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft,
        n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
        freeze_parameters=True)

    spectrogram_extractor.to(device)
    logmel_extractor.to(device)

    # Spectrogram
    pt_spectrogram = spectrogram_extractor.forward(pt_data[None, :])

    # Log mel spectrogram
    pt_logmel_spectrogram = logmel_extractor.forward(pt_spectrogram)

    # Uncomment for debug
    if True:
        debug(select='default', device=device)
        debug(select='dft', device=device)
        debug(select='stft', device=device)
        debug(select='logmel', device=device)
        debug(select='enframe', device=device)
