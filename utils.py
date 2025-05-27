import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
from typing import Optional, Tuple, Any

def plot_image(
    image: np.ndarray,
    factor: float = 1.0,
    clip_range: Optional[Tuple[float, float]] = None,
    **kwargs: Any
) -> None:
    """
    Plot an image array.
    
    Args:
        image (numpy.ndarray): Image array to plot
        factor (float): Multiplicative factor for image values
        clip_range (tuple): (min, max) range to clip values
        **kwargs: Additional arguments to pass to plt.imshow()
    """
    plt.figure(figsize=(15, 15))
    if clip_range is not None:
        image = np.clip(image * factor, *clip_range)
    else:
        image = image * factor
    plt.imshow(image, **kwargs)
    plt.axis('off')
    plt.show()

def plot_points(df: pd.DataFrame, filename: str = "all_points.html") -> pdk.Deck:
    layer = pdk.Layer(
        'ScatterplotLayer',
        df,
        get_position=['longitude', 'latitude'],
        auto_highlight=True,
        get_radius=1000,  # Radius is given in meters
        get_fill_color=[240, 0, 200, 140],
        pickable=True
    )

    view_state = pdk.ViewState(
        longitude=-70, 
        latitude=-10, 
        zoom=2,
    )

    # Render
    r = pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state,
        map_style=pdk.map_styles.CARTO_ROAD,
    )
    r.to_html(filename)
    return r