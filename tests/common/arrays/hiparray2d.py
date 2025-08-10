import unittest

from hip import hip
import numpy as np

from GPUSimulators.common.hip_check import hip_check
from GPUSimulators.common.arrays.hip.array2d import HIPArray2D


class TestHIPArray2D(unittest.TestCase):
    def setUp(self):
        # Array parameters
        self.nx = 1024
        self.ny = 1024
        self.x_halo = 2
        self.y_halo = 2

        self.shape = (self.nx, self.ny)
        self.extent = (self.x_halo, self.y_halo, self.nx, self.ny)
        self.dtype = np.float32()

        self.stream = hip_check(hip.hipStreamCreate())

    def tearDown(self):
        hip_check(hip.hipStreamDestroy(self.stream))

    def test_zeros_transfer(self):
        upload = np.zeros(shape=self.shape, dtype=self.dtype)

        array = HIPArray2D(stream=self.stream, nx=self.nx, ny=self.ny, x_halo=self.x_halo, y_halo=self.y_halo,
                           dtype=self.dtype)

        array.upload(stream=self.stream, cpu_data=upload, extent=self.extent)

        # Download to check the data is sane
        # Download without passing an array
        download = array.download(stream=self.stream, asynch=False, extent=self.extent)

        self.assertTrue(np.allclose(upload, download), "Download without cpu_data does not provide equal arrays.")

        # Download with passing an array
        download_array = np.zeros_like(upload)
        array.download(stream=self.stream, cpu_data=download_array, asynch=False, extent=self.extent)
        self.assertTrue(np.allclose(upload, download_array),
                        "Download with cpu_data does not provide equal arrays.")

    def test_int_transfer(self):
        upload = np.ones(shape=self.shape, dtype=self.dtype)
        array = HIPArray2D(stream=self.stream, nx=self.nx, ny=self.ny, x_halo=self.x_halo, y_halo=self.y_halo,
                           dtype=self.dtype)

        array.upload(stream=self.stream, cpu_data=upload, extent=self.extent)

        # Download to check the data is sane
        # Download without passing an array
        download = array.download(stream=self.stream, asynch=False, extent=self.extent)

        self.assertTrue(np.allclose(upload, download), "Download without cpu_data does not provide equal arrays.")

        # Download with passing an array
        download_array = np.zeros_like(upload)
        array.download(stream=self.stream, cpu_data=download_array, asynch=False, extent=self.extent)
        self.assertTrue(np.allclose(upload, download_array),
                        "Download with cpu_data does not provide equal arrays.")

    def test_random_transfer(self):
        rng = np.random.default_rng(seed=42)

        upload = rng.random(size=self.shape, dtype=self.dtype)

        array = HIPArray2D(stream=self.stream, nx=self.nx, ny=self.ny, x_halo=self.x_halo, y_halo=self.y_halo,
                           dtype=self.dtype)

        array.upload(stream=self.stream, cpu_data=upload, extent=self.extent)

        # Download to check the data is sane
        # Download without passing an array
        download = array.download(stream=self.stream, asynch=False, extent=self.extent)

        self.assertTrue(np.allclose(upload, download), "Download without cpu_data does not provide equal arrays.")

        # Download with passing an array
        download_array = np.zeros_like(upload)
        array.download(stream=self.stream, cpu_data=download_array, asynch=False, extent=self.extent)
        self.assertTrue(np.allclose(upload, download_array),
                        "Download with cpu_data does not provide equal arrays.")

    def test_async(self):
        rng = np.random.default_rng(seed=42)

        upload = rng.random(size=self.shape, dtype=self.dtype)
        array = HIPArray2D(stream=self.stream, nx=self.nx, ny=self.ny, x_halo=self.x_halo, y_halo=self.y_halo,
                           dtype=self.dtype)

        array.upload(stream=self.stream, cpu_data=upload, extent=self.extent)

        # Download without passing an array
        download = array.download(stream=self.stream, asynch=True, extent=self.extent)
        hip_check(hip.hipStreamSynchronize(self.stream))

        self.assertTrue(np.allclose(upload, download), "Download without cpu_data does not provide equal arrays.")

        # Redo test but passing an array this time
        upload = rng.random(size=self.shape, dtype=self.dtype)

        array.upload(stream=self.stream, cpu_data=upload, extent=self.extent)

        # Download by passing an array
        download_array = np.zeros_like(upload)
        array.download(stream=self.stream, cpu_data=download_array, asynch=True, extent=self.extent)
        hip_check(hip.hipStreamSynchronize(self.stream))

        self.assertTrue(np.allclose(upload, download_array),
                        "Download with cpu_data does not provide equal arrays.")



if __name__ == '__main__':
    unittest.main()
