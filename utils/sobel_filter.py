import math
import torch


class NormGradient(torch.nn.Module):
    def __init__(self, device, ignore_border=False, return_norm=True):
        """
        Creates an instance of a sobel filter in x or y direction

        :param direction: Can be 'x_direction' or 'y_direction'
        """
        super(NormGradient, self).__init__()
        self.ignore_border = ignore_border
        self.return_norm = return_norm

        # X- Camera coordinate system
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float)

        # Y- Camera coordinate system
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float)

        pad = 2 if self.ignore_border else 1

        self.padding_layer = torch.nn.ReflectionPad2d((pad, pad, pad, pad))
        self.filter_matrix_x = sobel_x.expand(1, 1, 3, 3)
        self.filter_matrix_y = sobel_y.expand(1, 1, 3, 3)

    def forward(self, x):
        if self.ignore_border:
            x = x[:, :, 1:-1, 1:-1]
        x = self.padding_layer(x)
        gradient_x = torch.nn.functional.conv2d(x, self.filter_matrix_x, padding=0, stride=1)
        gradient_y = torch.nn.functional.conv2d(x, self.filter_matrix_y, padding=0, stride=1)

        if self.return_norm:
            return torch.sqrt(gradient_x**2 + gradient_y**2)

        return torch.cat([gradient_x, gradient_y], dim=1)


class GaussianSmoothing(torch.nn.Module):
    """
    Adapted from:
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/9
    """
    def __init__(self, channels, kernel_size, sigma, device):
        super(GaussianSmoothing, self).__init__()
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32, device=device) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        self.kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.padding_layer = torch.nn.ReflectionPad2d((kernel_size[0]//2, kernel_size[1]//2,
                                                       kernel_size[0]//2, kernel_size[1]//2))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = self.padding_layer(input)

        return torch.nn.functional.conv2d(input, weight=self.kernel)
