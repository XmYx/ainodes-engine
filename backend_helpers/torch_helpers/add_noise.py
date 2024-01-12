import torch


def add_noise(image, noise_type='gaussian', noise_amount=0.05):
    """
    This function adds various types of noise to a PyTorch tensor image.

    Args:
        image (torch.Tensor): Input tensor image of size (C, H, W).
        noise_type (str): The type of noise to add. Must be one of 'gaussian', 'salt_pepper', or 'poisson'. ['gaussian', 'salt_pepper', 'poisson']
        noise_amount (float): The amount of noise to add. This should be a float value between 0 and 1.

    Returns:
        torch.Tensor: The image with added noise.

    Raises:
        ValueError: If an unsupported noise_type is provided.
    """

    if noise_type == 'gaussian':
        # Gaussian noise is simply random numbers drawn from a Gaussian distribution added onto the data.
        noise = torch.randn_like(image) * noise_amount
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0., 1.)  # make sure the values are still between [0, 1]

    elif noise_type == 'salt_pepper':
        # Salt-and-pepper noise presents as sparsely occurring white and black pixels.
        mask = torch.rand_like(image)
        noisy_image = image.clone()
        noisy_image[mask < noise_amount / 2] = 0.  # pepper noise
        noisy_image[mask > 1 - noise_amount / 2] = 1.  # salt noise

    elif noise_type == 'poisson':
        # Poisson noise is a type of noise which can be modeled by a Poisson process. It is the predominant form of noise in relatively bright parts of an image.
        # For this, we assume that the image has values between 0 and 1, and that the noise_amount represents the lambda (rate parameter) for the Poisson distribution.
        noise = torch.poisson(image * noise_amount) / noise_amount
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0., 1.)  # make sure the values are still between [0, 1]

    else:
        raise ValueError(
            f'Unsupported noise type: {noise_type}. Must be one of ["gaussian", "salt_pepper", "poisson"].')

    return noisy_image


def add_noise_with_mask(image, mask, noise_type='gaussian', noise_amount=0.05):
    """
    This function adds various types of noise to a PyTorch tensor image.

    Args:
        image (torch.Tensor): Input tensor image of size (C, H, W).
        mask (torch.Tensor): An RGBA tensor of the same size as 'image'. The alpha channel is used to
                             determine where the noise will be added.
        noise_type (str): The type of noise to add. Must be one of 'gaussian', 'salt_pepper', or 'poisson'.
        noise_amount (float): The amount of noise to add. This should be a float value between 0 and 1.

    Returns:
        torch.Tensor: The image with added noise.

    Raises:
        ValueError: If an unsupported noise_type is provided.
    """

    # Get the alpha channel and normalize it to be between 0 and 1.

    alpha_mask = mask / 255

    if noise_type == 'gaussian':
        noise = torch.randn_like(image) * noise_amount
        noisy_image = image + (noise * alpha_mask)
        noisy_image = torch.clamp(noisy_image, 0., 1.)  # make sure the values are still between [0, 1]

    elif noise_type == 'salt_pepper':
        random_noise = torch.rand_like(image)
        salt_pepper_noise = torch.clone(image)
        salt_pepper_noise[random_noise < noise_amount / 2] = 0.  # pepper noise
        salt_pepper_noise[random_noise > 1 - noise_amount / 2] = 1.  # salt noise
        noisy_image = image * (1 - alpha_mask) + salt_pepper_noise * alpha_mask  # apply noise only on mask regions

    elif noise_type == 'poisson':
        noise = torch.poisson(image * noise_amount) / noise_amount
        noisy_image = image + noise * alpha_mask
        noisy_image = torch.clamp(noisy_image, 0., 1.)  # make sure the values are still between [0, 1]

    else:
        raise ValueError(
            f'Unsupported noise type: {noise_type}. Must be one of ["gaussian", "salt_pepper", "poisson"].')

    return noisy_image