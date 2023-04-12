import numpy as np
import torch
import ssim as sm


def _check_picture(mode: str, img1, img2):
    if mode == "shape":
        assert img1.shape == img2.shape, "two pictures must have the same size."


def _to_numpy():
    pass


def get_iqa_list():
    return ["uqi", "mse", "snr", "psnr", "ssim", "ms_ssim", "nqm", "fsim"]


def fsim(image_origin, image_query):
    import torch
    from .fsim import FSIMc
    to_tensor = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
    image_origin = to_tensor(image_origin)
    image_query = to_tensor(image_query)
    _check_picture("shape", image_origin, image_query)
    img1b = torch.unsqueeze(image_origin, 0)
    img2b = torch.unsqueeze(image_query, 0)
    FSIM_loss = FSIMc()
    loss = FSIM_loss(img1b, img2b)
    return loss.item()


def uqi(image_origin, image_query, block_size=8):
    from scipy.signal import correlate2d

    """
    Input : an original image and a test image of the same size
    Output: (1) an overall quality index of the test image, with a value
                range of [-1, 1].
            (2) a quality map of the test image. The map has a smaller
                size than the input images. The actual size is
                img_size - BLOCK_SIZE + 1.

    Usage:

    1. Load the original and the test images into two matrices
       (say image_origin and image_query)

    2. Run this function in one of the two ways:

        Choice 1 (suggested):
       qi,qi_map = uqi(img1, img2)

        Choice 2:
       qi,qi_map = uqi(img1, img2, block_size)

       The default block_size is 8 (Choice 1). Otherwise, you can specify
       it by yourself (Choice 2).

    3. See the results:

       qi                    Gives the over quality index.
       imshow((qi_map+1)/2)  Shows the quality map as an image.
    """
    to_numpy = lambda x: np.array(x.convert("L"), dtype=np.float32)
    image_origin = to_numpy(image_origin)
    image_query = to_numpy(image_query)
    _check_picture("shape", image_origin, image_query)
    N = block_size ** 2
    sum2_filter = np.ones((block_size, block_size))

    img1_sq = image_origin * image_origin
    img2_sq = image_query * image_query
    img12 = image_origin * image_query

    img1_sum = correlate2d(image_origin, sum2_filter, "valid")
    img2_sum = correlate2d(image_query, sum2_filter, "valid")
    img1_sq_sum = correlate2d(img1_sq, sum2_filter, "valid")
    img2_sq_sum = correlate2d(img2_sq, sum2_filter, "valid")
    img12_sum = correlate2d(img12, sum2_filter, "valid")

    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum ** 2 + img2_sum ** 2

    numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul

    quality_map = np.ones(np.shape(denominator))
    index = (denominator1 == 0) & (img12_sq_sum_mul != 0)
    quality_map[index] = 2 * img12_sum_mul[index] / img12_sq_sum_mul[index]
    index = denominator != 0
    quality_map[index] = numerator[index] / denominator[index]
    quality = np.mean(quality_map)
    return float(quality)


def mse(image_origin, image_query):
    to_numpy = lambda x: np.array(x, dtype=float)
    image_origin = to_numpy(image_origin)
    image_query = to_numpy(image_query)
    _check_picture("shape", image_origin, image_query)
    return np.mean((image_origin - image_query) ** 2)


def snr(image_origin, image_query):
    to_numpy = lambda x: np.array(x, dtype=float)
    image_origin = to_numpy(image_origin)
    image_query = to_numpy(image_query)
    _check_picture("shape", image_origin, image_query)
    signal_value = np.mean(image_origin ** 2)
    mse_value = mse(image_origin, image_query)
    return 10 * np.log10(signal_value / mse_value)


def psnr(image_origin, image_query):
    to_numpy = lambda x: np.array(x, dtype=float)
    image_origin = to_numpy(image_origin)
    image_query = to_numpy(image_query)
    _check_picture("shape", image_origin, image_query)
    mse_value = mse(image_origin, image_query)
    return 10 * np.log10(255 * 255 / mse_value)


def ssim(image_origin, image_query):
    X = torch.tensor(np.array(image_origin), dtype=torch.float32)
    Y = torch.tensor(np.array(image_query), dtype=torch.float32)
    X = X.permute(2, 0, 1).unsqueeze(0)
    Y = Y.permute(2, 0, 1).unsqueeze(0)
    _check_picture("shape", X, Y)
    return sm.ssim(X, Y).item()


def ms_ssim(image_origin, image_query):
    X = torch.tensor(np.array(image_origin), dtype=torch.float32)
    Y = torch.tensor(np.array(image_query), dtype=torch.float32)
    X = X.permute(2, 0, 1).unsqueeze(0)
    Y = Y.permute(2, 0, 1).unsqueeze(0)
    _check_picture("shape", X, Y)
    return sm.ms_ssim(X, Y).item()


def nqm(image_origin, image_query, view_angle=1):
    from numpy import cos, log2, pi, real, log10
    from numpy.fft import fft2, fftshift, ifft2
    to_numpy = lambda x: np.array(x.convert("L"), dtype=np.float32)
    image_origin = to_numpy(image_origin)
    image_query = to_numpy(image_query)
    _check_picture("shape", image_origin, image_query)

    def ctf(f_r):
        if isinstance(f_r, float) or isinstance(f_r, int):
            y = 1. / (200 * (2.6 * (0.0192 + 0.114 * f_r) * np.exp(-(0.114 * (f_r)) ** 1.1)))
            return y
        s = np.shape(f_r)
        f_r = f_r.flatten()
        y = 1. / (200 * (2.6 * (0.0192 + 0.114 * f_r) * np.exp(-(0.114 * (f_r)) ** 1.1)))
        return y.reshape(np.shape(f_r))

    def find(condition):
        return np.nonzero(condition)

    def cmaskn_modified(c, ci, a, ai, i):
        H, W = np.shape(c)
        ci = ci.flatten()
        c = c.flatten()
        t = find(np.abs(ci) > 1)
        ci[t] = 1
        ai = ai.flatten()
        a = a.flatten()
        ct = ctf(i)
        T = ct * (.86 * ((c / ct) - 1) + .3)

        a1 = find((np.abs(ci - c) - T) < 0)

        ai[a1] = a[a1]

        return ai.reshape(H, W)

    def gthresh_modified(x, T, z):
        H, W = np.shape(x)
        x = x.flatten()
        z = z.flatten()
        a = find(np.abs(x) < T)

        z[a] = np.zeros(np.shape(a))

        return z.reshape(H, W)

    i = complex(0, 1)
    O = image_origin
    I = image_query
    VA = view_angle
    x, y = np.shape(O)
    xplane, yplane = np.meshgrid(np.arange(-y / 2, y / 2), np.arange(-x / 2, x / 2))
    plane = (xplane + i * yplane)
    r = np.abs(plane)
    FO = fft2(O)
    FI = fft2(I)
    G_0 = 0.5 * (
            1 + cos(pi * log2((r + 2) * ((r + 2 <= 4) * (r + 2 >= 1)) + 4 * (~((r + 2 <= 4) * (r + 2 >= 1)))) - pi))
    G_1 = 0.5 * (1 + cos(pi * log2(r * ((r <= 4) * (r >= 1)) + 4 * (~((r <= 4) * (r >= 1)))) - pi))
    G_2 = 0.5 * (1 + cos(pi * log2(r * ((r >= 2) * (r <= 8)) + .5 * (~((r >= 2) * (r <= 8))))))
    G_3 = 0.5 * (1 + cos(pi * log2(r * ((r >= 4) * (r <= 16)) + 4 * (~((r >= 4) * (r <= 16)))) - pi))
    G_4 = 0.5 * (1 + cos(pi * log2(r * ((r >= 8) * (r <= 32)) + .5 * (~((r >= 8) * (r <= 32))))))
    G_5 = 0.5 * (1 + cos(pi * log2(r * ((r >= 16) * (r <= 64)) + 4 * (~((r >= 16) * (r <= 64)))) - pi))
    GS_0 = fftshift(G_0)
    GS_1 = fftshift(G_1)
    GS_2 = fftshift(G_2)
    GS_3 = fftshift(G_3)
    GS_4 = fftshift(G_4)
    GS_5 = fftshift(G_5)

    L_0 = ((GS_0) * FO)
    LI_0 = (GS_0 * FI)

    l_0 = real(ifft2(L_0))
    li_0 = real(ifft2(LI_0))

    A_1 = GS_1 * FO
    AI_1 = (GS_1 * FI)

    a_1 = real(ifft2(A_1))
    ai_1 = real(ifft2(AI_1))

    A_2 = GS_2 * FO
    AI_2 = GS_2 * FI

    a_2 = real(ifft2(A_2))
    ai_2 = real(ifft2(AI_2))

    A_3 = GS_3 * FO
    AI_3 = GS_3 * FI

    a_3 = real(ifft2(A_3))
    ai_3 = real(ifft2(AI_3))

    A_4 = GS_4 * FO
    AI_4 = GS_4 * FI

    a_4 = real(ifft2(A_4))
    ai_4 = real(ifft2(AI_4))

    A_5 = GS_5 * FO
    AI_5 = GS_5 * FI

    a_5 = real(ifft2(A_5))
    ai_5 = real(ifft2(AI_5))
    del FO
    del FI

    del G_0
    del G_1
    del G_2
    del G_3
    del G_4
    del G_5

    del GS_0
    del GS_1
    del GS_2
    del GS_3
    del GS_4
    del GS_5
    c1 = ((a_1 / (l_0)))
    c2 = (a_2 / (l_0 + a_1))
    c3 = (a_3 / (l_0 + a_1 + a_2))
    c4 = (a_4 / (l_0 + a_1 + a_2 + a_3))
    c5 = (a_5 / (l_0 + a_1 + a_2 + a_3 + a_4))

    ci1 = (ai_1 / (li_0))
    ci2 = (ai_2 / (li_0 + ai_1))
    ci3 = (ai_3 / (li_0 + ai_1 + ai_2))
    ci4 = (ai_4 / (li_0 + ai_1 + ai_2 + ai_3))
    ci5 = (ai_5 / (li_0 + ai_1 + ai_2 + ai_3 + ai_4))

    d1 = ctf(2 / VA)
    d2 = ctf(4 / VA)
    d3 = ctf(8 / VA)
    d4 = ctf(16 / VA)
    d5 = ctf(32 / VA)

    ai_1 = cmaskn_modified(c1, ci1, a_1, ai_1, 1)
    ai_2 = cmaskn_modified(c2, ci2, a_2, ai_2, 2)
    ai_3 = cmaskn_modified(c3, ci3, a_3, ai_3, 3)
    ai_4 = cmaskn_modified(c4, ci4, a_4, ai_4, 4)
    ai_5 = cmaskn_modified(c5, ci5, a_5, ai_5, 5)

    A_1 = gthresh_modified(c1, d1, a_1)
    AI_1 = gthresh_modified(ci1, d1, ai_1)
    A_2 = gthresh_modified(c2, d2, a_2)
    AI_2 = gthresh_modified(ci2, d2, ai_2)
    A_3 = gthresh_modified(c3, d3, a_3)
    AI_3 = gthresh_modified(ci3, d3, ai_3)
    A_4 = gthresh_modified(c4, d4, a_4)
    AI_4 = gthresh_modified(ci4, d4, ai_4)
    A_5 = gthresh_modified(c5, d5, a_5)
    AI_5 = gthresh_modified(ci5, d5, ai_5)

    y1 = (A_1 + A_2 + A_3 + A_4 + A_5)
    y2 = (AI_1 + AI_2 + AI_3 + AI_4 + AI_5)

    square_err = (y1 - y2) * (y1 - y2)
    bp = sum(sum(square_err))

    sp = sum(sum(y1 ** 2))

    return 10 * log10(sp / bp)
