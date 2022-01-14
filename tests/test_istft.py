import pytest
import torch
import torch.testing
from numpy.testing import assert_array_almost_equal

from torchlibrosa import ISTFT, STFT


class TestISTFT:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_move_onnx_compatible_istft_between_devices(self):
        # Check that the ONNX variant of ISTFT works when initialized with device="cuda"
        # and that it works as expected when moved to CPU afterwards.
        length = 16000
        waveform = torch.rand(1, length, dtype=torch.float32, device="cuda")

        stft = STFT().cuda()

        (real, imag) = stft(waveform)
        frames_num = real.shape[2]

        istft = ISTFT(
            onnx=True,
            frames_num=frames_num,
            device="cuda",
        ).cuda()

        waveform_output_cuda = istft(real, imag, length=length)

        # Move to CPU and do the pass there
        real = real.cpu()
        imag = imag.cpu()
        istft = istft.cpu()
        waveform_output_cpu = istft(real, imag, length=length)

        assert_array_almost_equal(
            waveform_output_cpu.numpy(), waveform_output_cuda.cpu().numpy(), decimal=5
        )
