"""PyTorch layer for extracting image features for the film_net interpolator.

The feature extractor implemented here converts an image pyramid into a pyramid
of deep features. The feature pyramid serves a similar purpose as U-Net
architecture's encoder, but we use a special cascaded architecture described in
Multi-view Image Fusion [1].

For comprehensiveness, below is a short description of the idea. While the
description is a bit involved, the cascaded feature pyramid can be used just
like any image feature pyramid.

Why cascaded architeture?
=========================
To understand the concept it is worth reviewing a traditional feature pyramid
first: *A traditional feature pyramid* as in U-net or in many optical flow
networks is built by alternating between convolutions and pooling, starting
from the input image.

It is well known that early features of such architecture correspond to low
level concepts such as edges in the image whereas later layers extract
semantically higher level concepts such as object classes etc. In other words,
the meaning of the filters in each resolution level is different. For problems
such as semantic segmentation and many others this is a desirable property.

However, the asymmetric features preclude sharing weights across resolution
levels in the feature extractor itself and in any subsequent neural networks
that follow. This can be a downside, since optical flow prediction, for
instance is symmetric across resolution levels. The cascaded feature
architecture addresses this shortcoming.

How is it built?
================
The *cascaded* feature pyramid contains feature vectors that have constant
length and meaning on each resolution level, except few of the finest ones. The
advantage of this is that the subsequent optical flow layer can learn
synergically from many resolutions. This means that coarse level prediction can
benefit from finer resolution training examples, which can be useful with
moderately sized datasets to avoid overfitting.

The cascaded feature pyramid is built by extracting shallower subtree pyramids,
each one of them similar to the traditional architecture. Each subtree
pyramid S_i is extracted starting from each resolution level:

image resolution 0 -> S_0
image resolution 1 -> S_1
image resolution 2 -> S_2
...

If we denote the features at level j of subtree i as S_i_j, the cascaded pyramid
is constructed by concatenating features as follows (assuming subtree depth=3):

lvl
feat_0 = concat(                               S_0_0 )
feat_1 = concat(                         S_1_0 S_0_1 )
feat_2 = concat(                   S_2_0 S_1_1 S_0_2 )
feat_3 = concat(             S_3_0 S_2_1 S_1_2       )
feat_4 = concat(       S_4_0 S_3_1 S_2_2             )
feat_5 = concat( S_5_0 S_4_1 S_3_2                   )
   ....

In above, all levels except feat_0 and feat_1 have the same number of features
with similar semantic meaning. This enables training a single optical flow
predictor module shared by levels 2,3,4,5... . For more details and evaluation
see [1].

[1] Multi-view Image Fusion, Trinidad et al. 2019
"""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from .film_util import conv


class SubTreeExtractor(nn.Module):
    """Extracts a hierarchical set of features from an image.

    This is a conventional, hierarchical image feature extractor, that extracts
    [k, k*2, k*4... ] filters for the image pyramid where k=options.sub_levels.
    Each level is followed by average pooling.
    """

    def __init__(self, in_channels=3, channels=64, n_layers=4):
        super().__init__()
        convs = []
        for i in range(n_layers):
            convs.append(nn.Sequential(
                conv(in_channels, (channels << i), 3),
                conv((channels << i), (channels << i), 3)
            ))
            in_channels = channels << i
        self.convs = nn.ModuleList(convs)

    def forward(self, image: torch.Tensor, n: int) -> List[torch.Tensor]:
        """Extracts a pyramid of features from the image.

        Args:
          image: TORCH.Tensor with shape BATCH_SIZE x HEIGHT x WIDTH x CHANNELS.
          n: number of pyramid levels to extract. This can be less or equal to
           options.sub_levels given in the __init__.
        Returns:
          The pyramid of features, starting from the finest level. Each element
          contains the output after the last convolution on the corresponding
          pyramid level.
        """
        head = image
        pyramid = []
        for i, layer in enumerate(self.convs):
            head = layer(head)
            pyramid.append(head)
            if i < n - 1:
                head = F.avg_pool2d(head, kernel_size=2, stride=2)
        return pyramid


class FeatureExtractor(nn.Module):
    """Extracts features from an image pyramid using a cascaded architecture.
    """

    def __init__(self, in_channels=3, channels=64, sub_levels=4):
        super().__init__()
        self.extract_sublevels = SubTreeExtractor(in_channels, channels, sub_levels)
        self.sub_levels = sub_levels

    def forward(self, image_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        """Extracts a cascaded feature pyramid.

        Args:
          image_pyramid: Image pyramid as a list, starting from the finest level.
        Returns:
          A pyramid of cascaded features.
        """
        sub_pyramids: List[List[torch.Tensor]] = []
        for i in range(len(image_pyramid)):
            # At each level of the image pyramid, creates a sub_pyramid of features
            # with 'sub_levels' pyramid levels, re-using the same SubTreeExtractor.
            # We use the same instance since we want to share the weights.
            #
            # However, we cap the depth of the sub_pyramid so we don't create features
            # that are beyond the coarsest level of the cascaded feature pyramid we
            # want to generate.
            capped_sub_levels = min(len(image_pyramid) - i, self.sub_levels)
            sub_pyramids.append(self.extract_sublevels(image_pyramid[i], capped_sub_levels))
        # Below we generate the cascades of features on each level of the feature
        # pyramid. Assuming sub_levels=3, The layout of the features will be
        # as shown in the example on file documentation above.
        feature_pyramid: List[torch.Tensor] = []
        for i in range(len(image_pyramid)):
            features = sub_pyramids[i][0]
            for j in range(1, self.sub_levels):
                if j <= i:
                    features = torch.cat([features, sub_pyramids[i - j][j]], dim=1)
            feature_pyramid.append(features)
        return feature_pyramid
