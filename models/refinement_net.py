import torch
import torch.nn as nn

from utils.sobel_filter import NormGradient


class StyleRefinementNetwork(nn.Module):
    """Upsamples the latent space to the original dimensions."""
    def __init__(self, input_c, output_c, sensor='rgb', channel_list=(16, 8), last_layer_pad=0, device='cpu'):
        super(StyleRefinementNetwork, self).__init__()
        layer_in_c = input_c + 2
        self.nr_steps = 10
        self.max_disp = 10

        sequence = []
        for nr_channels in channel_list:
            layer_out_c = nr_channels
            sequence.append(nn.Conv2d(layer_in_c, layer_out_c, kernel_size=3, stride=1, padding=1, bias=False))
            sequence.append(nn.BatchNorm2d(layer_out_c, momentum=0.01))
            sequence.append(nn.LeakyReLU())
            layer_in_c = layer_out_c

        sequence.append(nn.Conv2d(layer_in_c, output_c, kernel_size=1, stride=1, padding=0, bias=False))

        if sensor == 'events' or sensor == 'rgb':
            sequence.append(nn.ReLU())

        self.noise_model = nn.Sequential(*sequence)
        self.norm_gradient_layer = NormGradient(device, ignore_border=True, return_norm=False)
        self.step_vector = (torch.arange(start=1, end=self.nr_steps+1, dtype=torch.float,
                                         device=device) / self.nr_steps)[None, None, :, None, None]

        self.blur_area = None
        self.sigmoid_layer = nn.Sigmoid()
        self.index_map = None

    def forward(self, decoder_output, img=None, noise_tensor=None, return_clean_reconst=False, return_flow=False):
        flow_map = decoder_output[:, :2, :, :]
        batch_s, _, height, width = flow_map.shape
        if noise_tensor is None:
            random_input = torch.cat([torch.randn(size=[batch_s, 1, height, width], device=flow_map.device),
                                      torch.randn(size=[batch_s, 1, 1, 1],
                                                  device=flow_map.device).expand(-1, -1, height, width)],
                                     dim=1)
        else:
            random_input = noise_tensor

        clean_event_histogram = self.generate_events(flow_map, img)

        x_noise = torch.zeros_like(clean_event_histogram)

        x = torch.cat([clean_event_histogram.detach(), random_input], dim=1)
        x_noise += self.noise_model(x)

        if return_clean_reconst and return_flow:
            return x_noise + clean_event_histogram, clean_event_histogram, flow_map

        if return_clean_reconst:
            return x_noise + clean_event_histogram, clean_event_histogram

        return x_noise + clean_event_histogram

    def generate_events(self, flow_map, img):
        batch_s, _, height, width = flow_map.shape
        # Gradient map is expressed in the camera coordinate system: (x, y) = (u, v)
        gradient_map = self.norm_gradient_layer.forward(img)
        generated_events = -(gradient_map * flow_map).sum(1, keepdim=True)
        clean_event_histogram = torch.zeros([batch_s, 2, height, width], device=generated_events.device)
        pos_event_mask = (generated_events > 0).int()
        clean_event_histogram[:, 0, :, :] = -(generated_events * (1 - pos_event_mask))[:, 0, :, :]
        clean_event_histogram[:, 1, :, :] = (generated_events * pos_event_mask)[:, 0, :, :]

        return clean_event_histogram

    def check_boundaries(self, tensor, height, width):
        tensor[:, 0] = torch.clamp(tensor[:, 0], min=0, max=width-1)
        tensor[:, 1] = torch.clamp(tensor[:, 1], min=0, max=height-1)
        return tensor
