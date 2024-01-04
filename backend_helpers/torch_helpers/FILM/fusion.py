"""The final fusion stage for the film_net frame interpolator.

The inputs to this module are the warped input images, image features and
flow fields, all aligned to the target frame (often midway point between the
two original inputs). The output is the final image. FILM has no explicit
occlusion handling -- instead using the abovementioned information this module
automatically decides how to best blend the inputs together to produce content
in areas where the pixels can only be borrowed from one of the inputs.

Similarly, this module also decides on how much to blend in each input in case
of fractional timestep that is not at the halfway point. For example, if the two
inputs images are at t=0 and t=1, and we were to synthesize a frame at t=0.1,
it often makes most sense to favor the first input. However, this is not
always the case -- in particular in occluded pixels.

The architecture of the Fusion module follows U-net [1] architecture's decoder
side, e.g. each pyramid level consists of concatenation with upsampled coarser
level output, and two 3x3 convolutions.

The upsampling is implemented as 'resize convolution', e.g. nearest neighbor
upsampling followed by 2x2 convolution as explained in [2]. The classic U-net
uses max-pooling which has a tendency to create checkerboard artifacts.

[1] Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image
    Segmentation, 2015, https://arxiv.org/pdf/1505.04597.pdf
[2] https://distill.pub/2016/deconv-checkerboard/
"""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from .film_util import conv

_NUMBER_OF_COLOR_CHANNELS = 3


def get_channels_at_level(level, filters):
    n_images = 2
    channels = _NUMBER_OF_COLOR_CHANNELS
    flows = 2

    return (sum(filters << i for i in range(level)) + channels + flows) * n_images


class Fusion(nn.Module):
    """The decoder."""

    def __init__(self, n_layers=4, specialized_layers=3, filters=64):
        """
        Args:
            m: specialized levels
        """
        super().__init__()

        # The final convolution that outputs RGB:
        self.output_conv = nn.Conv2d(filters, 3, kernel_size=1)

        # Each item 'convs[i]' will contain the list of convolutions to be applied
        # for pyramid level 'i'.
        self.convs = nn.ModuleList()

        # Create the convolutions. Roughly following the feature extractor, we
        # double the number of filters when the resolution halves, but only up to
        # the specialized_levels, after which we use the same number of filters on
        # all levels.
        #
        # We create the convs in fine-to-coarse order, so that the array index
        # for the convs will correspond to our normal indexing (0=finest level).
        # in_channels: tuple = (128, 202, 256, 522, 512, 1162, 1930, 2442)

        in_channels = get_channels_at_level(n_layers, filters)
        increase = 0
        for i in range(n_layers)[::-1]:
            num_filters = (filters << i) if i < specialized_layers else (filters << specialized_layers)
            convs = nn.ModuleList([
                conv(in_channels, num_filters, size=2, activation=None),
                conv(in_channels + (increase or num_filters), num_filters, size=3),
                conv(num_filters, num_filters, size=3)]
            )
            self.convs.append(convs)
            in_channels = num_filters
            increase = get_channels_at_level(i, filters) - num_filters // 2

    def forward(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """Runs the fusion module.

        Args:
          pyramid: The input feature pyramid as list of tensors. Each tensor being
            in (B x H x W x C) format, with finest level tensor first.

        Returns:
          A batch of RGB images.
        Raises:
          ValueError, if len(pyramid) != config.fusion_pyramid_levels as provided in
            the constructor.
        """

        # As a slight difference to a conventional decoder (e.g. U-net), we don't
        # apply any extra convolutions to the coarsest level, but just pass it
        # to finer levels for concatenation. This choice has not been thoroughly
        # evaluated, but is motivated by the educated guess that the fusion part
        # probably does not need large spatial context, because at this point the
        # features are spatially aligned by the preceding warp.
        net = pyramid[-1]

        # Loop starting from the 2nd coarsest level:
        # for i in reversed(range(0, len(pyramid) - 1)):
        for k, layers in enumerate(self.convs):
            i = len(self.convs) - 1 - k
            # Resize the tensor from coarser level to match for concatenation.
            level_size = pyramid[i].shape[2:4]
            net = F.interpolate(net, size=level_size, mode='nearest')
            net = layers[0](net)
            net = torch.cat([pyramid[i], net], dim=1)
            net = layers[1](net)
            net = layers[2](net)
        net = self.output_conv(net)
        return net
