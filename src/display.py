import math
import matplotlib.pyplot as plt
import numpy as np

def plot_image_grid(
    images,
    max_images=None,
    img_range="neg1_1",  # "neg1_1" or "0_1"
    base_img_size=0.9    # controls compactness
):
    """
    images: list or tensor of shape [N, 1, C, H, W] or [N, C, H, W]
    """
    if max_images is not None:
        images = images[:max_images]

    n = len(images)
    if n == 0:
        return

    # Automatically choose grid close to square
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig_w = cols * base_img_size
    fig_h = rows * base_img_size

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]

            # Handle shape
            if img.ndim == 4:  # [1, C, H, W]
                img = img[0]

            img = np.transpose(img, (1, 2, 0))

            if img_range == "neg1_1":
                img = np.clip((img + 1) / 2, 0, 1)

            ax.imshow(img)
        ax.axis("off")

    # Ultra-tight spacing
    plt.subplots_adjust(
        left=0.01, right=0.99,
        top=0.99, bottom=0.01,
        wspace=0.01, hspace=0.01
    )

    plt.show()
