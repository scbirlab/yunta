"""Plotting functions."""

from typing import Optional

import os

from carabiner import print_err
from carabiner.mpl import colorblind_palette, figsaver, grid
from pandas import DataFrame
import numpy as np

def plot_matrix(
    m: np.ndarray, 
    filename_prefix: Optional[str] = None,
    format: str = 'png',
    dpi: int = 300, 
    vline: Optional[float] = None, 
    hline: Optional[float] = None,
    vmax: Optional[float] = None,
    *args, **kwargs
):
    
    fig, axes = grid()
    im = axes.imshow(m, cmap='magma', vmin=0., vmax=vmax)
    fig.colorbar(im, shrink=.7)
    if hline is not None:
        axes.axhline(hline, color='lightgrey', zorder=10)
    if vline is not None:
        axes.axvline(vline, color='lightgrey', zorder=10)
    axes.set(*args, **kwargs)

    if filename_prefix is not None:
        plot_dir = os.path.dirname(filename_prefix)
        if not os.path.exists(plot_dir):
            print_err(f"Creating output directory {plot_dir}")
            os.makedirs(plot_dir)
        df = DataFrame(
            m, 
            index=np.arange(m.shape[0]), 
            columns=np.arange(m.shape[1]),
        )
        figsaver(
            format=format,
            dpi=dpi,
        )(fig, filename_prefix, df=df)

    return fig, axes

def plot_dca(
    dca: np.ndarray, 
    filename_prefix: Optional[str] = None,
    format: str = 'png',
    dpi: int = 300
):
    
    if filename_prefix is not None:
        filename_prefix += "_dca"

    return plot_matrix(dca, filename_prefix=filename_prefix, format=format, dpi=dpi)