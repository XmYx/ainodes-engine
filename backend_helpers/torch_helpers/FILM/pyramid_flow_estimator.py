"""PyTorch layer for estimating optical flow by a residual flow pyramid.

This approach of estimating optical flow between two images can be traced back
to [1], but is also used by later neural optical flow computation methods such
as SpyNet [2] and PWC-Net [3].

The basic idea is that the optical flow is first estimated in a coarse
resolution, then the flow is upsampled to warp the higher resolution image and
then a residual correction is computed and added to the estimated flow. This
process is repeated in a pyramid on coarse to fine order to successively
increase the resolution of both optical flow and the warped image.

In here, the optical flow predictor is used as an internal component for the
film_net frame interpolator, to warp the two input images into the inbetween,
target frame.

[1] F. Glazer, Hierarchical motion detection. PhD thesis, 1987.
[2] A. Ranjan and M. J. Black, Optical Flow Estimation using a Spatial Pyramid
    Network. 2016
[3] D. Sun X. Yang, M-Y. Liu and J. Kautz, PWC-Net: CNNs for Optical Flow Using
    Pyramid, Warping, and Cost Volume, 2017
"""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

import film_util


class FlowEstimator(nn.Module):
    """Small-receptive field predictor for computing the flow between two images.

    This is used to compute the residual flow fields in PyramidFlowEstimator.

    Note that while the number of 3x3 convolutions & filters to apply is
    configurable, two extra 1x1 convolutions are appended to extract the flow in
    the end.

    Attributes:
      name: The name of the layer
      num_convs: Number of 3x3 convolutions to apply
      num_filters: Number of filters in each 3x3 convolution
    """

    def __init__(self, in_channels: int, num_convs: int, num_filters: int):
        super(FlowEstimator, self).__init__()

        self._convs = nn.ModuleList()
        for i in range(num_convs):
            self._convs.append(film_util.conv(in_channels=in_channels, out_channels=num_filters, size=3))
            in_channels = num_filters
        self._convs.append(film_util.conv(in_channels, num_filters // 2, size=1))
        in_channels = num_filters // 2
        # For the final convolution, we want no activation at all to predict the
        # optical flow vector values. We have done extensive testing on explicitly
        # bounding these values using sigmoid, but it turned out that having no
        # activation gives better results.
        self._convs.append(film_util.conv(in_channels, 2, size=1, activation=None))

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """Estimates optical flow between two images.

        Args:
          features_a: per pixel feature vectors for image A (B x H x W x C)
          features_b: per pixel feature vectors for image B (B x H x W x C)

        Returns:
          A tensor with optical flow from A to B
        """
        net = torch.cat([features_a, features_b], dim=1)
        for conv in self._convs:
            net = conv(net)
        return net


class PyramidFlowEstimator(nn.Module):
    """Predicts optical flow by coarse-to-fine refinement.
    """

    def __init__(self, filters: int = 64,
                 flow_convs: tuple = (3, 3, 3, 3),
                 flow_filters: tuple = (32, 64, 128, 256)):
        super(PyramidFlowEstimator, self).__init__()

        in_channels = filters << 1
        predictors = []
        for i in range(len(flow_convs)):
            predictors.append(
                FlowEstimator(
                    in_channels=in_channels,
                    num_convs=flow_convs[i],
                    num_filters=flow_filters[i]))
            in_channels += filters << (i + 2)
        self._predictor = predictors[-1]
        self._predictors = nn.ModuleList(predictors[:-1][::-1])

    def forward(self, feature_pyramid_a: List[torch.Tensor],
                feature_pyramid_b: List[torch.Tensor]) -> List[torch.Tensor]:
        """Estimates residual flow pyramids between two image pyramids.

        Each image pyramid is represented as a list of tensors in fine-to-coarse
        order. Each individual image is represented as a tensor where each pixel is
        a vector of image features.

        film_util.flow_pyramid_synthesis can be used to convert the residual flow
        pyramid returned by this method into a flow pyramid, where each level
        encodes the flow instead of a residual correction.

        Args:
          feature_pyramid_a: image pyramid as a list in fine-to-coarse order
          feature_pyramid_b: image pyramid as a list in fine-to-coarse order

        Returns:
          List of flow tensors, in fine-to-coarse order, each level encoding the
          difference against the bilinearly upsampled version from the coarser
          level. The coarsest flow tensor, e.g. the last element in the array is the
          'DC-term', e.g. not a residual (alternatively you can think of it being a
          residual against zero).
        """
        levels = len(feature_pyramid_a)
        v = self._predictor(feature_pyramid_a[-1], feature_pyramid_b[-1])
        residuals = [v]
        for i in range(levels - 2, len(self._predictors) - 1, -1):
            # Upsamples the flow to match the current pyramid level. Also, scales the
            # magnitude by two to reflect the new size.
            level_size = feature_pyramid_a[i].shape[2:4]
            v = F.interpolate(2 * v, size=level_size, mode='bilinear')
            # Warp feature_pyramid_b[i] image based on the current flow estimate.
            warped = film_util.warp(feature_pyramid_b[i], v)
            # Estimate the residual flow between pyramid_a[i] and warped image:
            v_residual = self._predictor(feature_pyramid_a[i], warped)
            residuals.insert(0, v_residual)
            v = v_residual + v

        for k, predictor in enumerate(self._predictors):
            i = len(self._predictors) - 1 - k
            # Upsamples the flow to match the current pyramid level. Also, scales the
            # magnitude by two to reflect the new size.
            level_size = feature_pyramid_a[i].shape[2:4]
            v = F.interpolate(2 * v, size=level_size, mode='bilinear')
            # Warp feature_pyramid_b[i] image based on the current flow estimate.
            warped = film_util.warp(feature_pyramid_b[i], v)
            # Estimate the residual flow between pyramid_a[i] and warped image:
            v_residual = predictor(feature_pyramid_a[i], warped)
            residuals.insert(0, v_residual)
            v = v_residual + v
        return residuals
