

import colorsys

from PIL import Image
from sklearn.cluster import KMeans
import wcag_contrast_ratio as contrast


def get_dominant_colours(path, *, count=5):
    """
    Return a list of the dominant RGB colours in the image at ``path``.

    :param path: Path to the image file.
    :param count: Number of dominant colours to find.

    """
    im = Image.open(path)

    # Resizing means less pixels to handle, so the *k*-means clustering converges
    # faster.  Small details are lost, but the main details will be preserved.
    im = im.resize((100, 100))

    # Ensure the image is RGB, and use RGB values in [0, 1] for consistency
    # with operations elsewhere.
    im = im.convert("RGB")
    colors = [(r / 255, g / 255, b / 255) for (r, g, b) in im.getdata()]

    return KMeans(n_clusters=count).fit(colors).cluster_centers_

def coloured_square(hex_string):
    """
    Returns a coloured square that you can print to a terminal.
    """
    hex_string = hex_string.strip("#")
    assert len(hex_string) == 6
    red = int(hex_string[:2], 16)
    green = int(hex_string[2:4], 16)
    blue = int(hex_string[4:6], 16)

    return f"\033[48:2::{red}:{green}:{blue}m \033[49m"


# example tint_color = choose_tint_color(dominant_colors, background_color=(0, 0, 0))
def choose_tint_color(dominant_colors, background_color):
    # The minimum contrast ratio for text and background to meet WCAG AA
    # is 4.5:1, so discard any dominant colours with a lower contrast.
    sufficient_contrast_colors = [
        col
        for col in dominant_colors
        if contrast.rgb(col, background_color) >= 4.5
    ]

    # If none of the dominant colours meet WCAG AA with the background,
    # try again with black and white -- every colour in the RGB space
    # has a contrast ratio of 4.5:1 with at least one of these, so we'll
    # get a tint colour, even if it's not a good one.
    #
    # Note: you could modify the dominant colours until one of them
    # has sufficient contrast, but that's omitted here because it adds
    # a lot of complexity for a relatively unusual case.
    if not sufficient_contrast_colors:
        return choose_tint_color(
            dominant_colors=dominant_colors + [(0, 0, 0), (1, 1, 1)],
            background_color=background_color
        )

    # Of the colours with sufficient contrast, pick the one with the
    # closest brightness (in the HSV colour space) to the background
    # colour.  This means we don't get very dark or very light colours,
    # but more bright, vibrant colours.
    hsv_background = colorsys.rgb_to_hsv(*background_color)
    hsv_candidates = {
        tuple(rgb_col): colorsys.rgb_to_hsv(*rgb_col)
        for rgb_col in sufficient_contrast_colors
    }

    candidates_by_brightness_diff = {
        rgb_col: abs(hsv_col[2] - hsv_background[2])
        for rgb_col, hsv_col in hsv_candidates.items()
    }

    rgb_choice, _ = min(
        candidates_by_brightness_diff.items(),
        key=lambda t: t[1]
    )

    assert rgb_choice in dominant_colors
    return rgb_choice



# if __name__ == "__main__":
#     import sys

#     try:
#         path = sys.argv[1]
#     except ImportError:
#         sys.exit(f"Usage: {__file__} <PATH>")

#     dominant_colors = get_dominant_colours(path, count=5)
#     tint_color = choose_tint_color(dominant_colors, background_color=(0, 0, 0))

#     print(tint_color)
#     print("#%02x%02x%02x" % tuple(int(v * 255) for v in tint_color))
# # %%
