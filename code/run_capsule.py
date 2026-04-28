import argparse
import glob
import itertools
import json
import logging
import os
import shutil
from datetime import datetime as dt
from joblib import Parallel, delayed
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
from aind_data_schema.core.data_description import DerivedDataDescription
from aind_data_schema.core.processing import (
    DataProcess,
    PipelineProcess,
    Processing,
    ProcessName,
)
from aind_data_schema.core.quality_control import (
    QCEvaluation,
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_metadata_upgrader.data_description_upgrade import DataDescriptionUpgrade
from aind_metadata_upgrader.processing_upgrade import ProcessingUpgrade
from aind_logging import setup_logging
from hdmf_zarr import NWBZarrIO
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.signal import butter, sosfiltfilt, welch

import utils.nwb_dict_utils as nwb_utils
from utils.preprocess import chunk_processing, motion_correct

"""
This capsule takes in an NWB file containing raw fiber photometry data
then process each channel (usually 4) of each ROI (usually 4) by
generating baseline-corrected (ΔF/F) and motion-corrected traces,
which are then appended back to the NWB file.
"""


def setup_logging_from_metadata(fiber_path: Path):
    """Setup logging from subject and data_description metadata.

    Parameters
    ----------
    fiber_path : Path
        Path to directory containing metadata files.
    """
    # Load data description
    data_description_path = fiber_path / "data_description.json"
    with open(data_description_path, "r") as f:
        data_description = json.load(f)

    process_name = os.getenv("PROCESS_NAME")
    asset_name = data_description.get("name")
    setup_logging(
        process_name,
        acquisition_name=asset_name,
        process_name=process_name,
        pipeline_name=os.getenv("PIPELINE_NAME", "")
    )


def write_output_metadata(
    metadata: dict,
    json_dir: Union[str, Path],
    process_name: Union[str, None],
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    start_date_time: dt,
) -> None:
    """Writes output metadata to processing.json

    Parameters
    ----------
    metadata : dict
        Parameters passed to the capsule.
    json_dir : Union[str, Path]
        Directory where the processing.json and data_description.json file is located.
    process_name : str
        Name of the process being recorded.
    input_fp : Union[str, Path]
        Path to the data input.
    output_fp : Union[str, Path]
        Path to the data output.
    start_date_time : dt
        Start date and time of the process.
    """
    proc_path = Path(json_dir) / "processing.json"

    dp = (
        [
            DataProcess(
                name=process_name,
                software_version=os.getenv("VERSION", ""),
                start_date_time=start_date_time,
                end_date_time=dt.now(),
                input_location=str(input_fp),
                output_location=str(output_fp),
                code_url=(os.getenv("DFF_EXTRACTION_URL")),
                parameters=metadata,
            )
        ]
        if process_name is not None
        else []
    )

    if proc_path.exists():
        with open(proc_path, "r") as f:
            proc_data = json.load(f)

        proc_upgrader = ProcessingUpgrade(old_processing_model=proc_data)
        processing = proc_upgrader.upgrade(
            processor_full_name="Fiberphotometry Processing Pipeline"
        )
        p = processing.processing_pipeline
        p.data_processes += dp
    else:
        p = PipelineProcess(
            processor_full_name="Fiberphotometry Processing Pipeline",
            data_processes=dp,
        )
        processing = Processing(processing_pipeline=p)
    if u := os.getenv("PIPELINE_URL", ""):
        p.pipeline_url = u
    if v := os.getenv("PIPELINE_VERSION", ""):
        p.pipeline_version = v
    processing.write_standard_file(output_directory=Path(output_fp))

    dd_file = Path(json_dir) / "data_description.json"
    if dd_file.exists():
        with open(dd_file, "r") as f:
            dd_data = json.load(f)
        dd_data["modality"] = [
            m
            for m in dd_data.get("modality", [])
            if isinstance(m, dict) and m.get("abbreviation") == "fib"
        ]
        dd_upgrader = DataDescriptionUpgrade(old_data_description_dict=dd_data)
        new_dd = dd_upgrader.upgrade()
        derived_dd = DerivedDataDescription.from_data_description(
            data_description=new_dd, process_name="processed"
        )
        derived_dd.write_standard_file(output_directory=Path(output_fp))
    else:
        logging.error("no input data description")


def plot_raw_dff_mc(
    nwb_file: pynwb.NWBFile,
    fiber: str,
    channels: list[str],
    method: str,
    fig_path: Path = Path("/results/dff-qc/"),
):
    """Plot raw, dF/F, and preprocessed (dF/F with motion correction) photometry traces
    for multiple channels from an NWB file.

    Parameters
    ----------
    nwb_file : NWBFile
        The Neurodata Without Borders (NWB) file containing photometry signal traces
        and their associated metadata.
    fiber : str
        The name of the fiber for which the signals should be plotted.
    channels : list of str
        A list of channel names to be plotted (e.g., ['G', 'R', 'Iso']).
    method : str
        The name of the preprocessing method used ("poly", "exp", or "bright").
    fig_path : Path, optional
        The path where the generated plot will be saved. Defaults to "/results/dff-qc/".
    """
    fig, ax = plt.subplots(3, 1, figsize=(12, 4), sharex=True)
    for i, suffix in enumerate(("", f"_dff-{method}", f"_dff-{method}_mc-iso-IRLS")):
        for ch in sorted(channels):
            if i == 0:
                trace = nwb_file.acquisition[ch + f"_{fiber}"]
            else:
                trace = nwb_file.processing["fiber_photometry"].data_interfaces[
                    ch + f"_{fiber}{suffix}"
                ]
            t, d = trace.timestamps[:], trace.data[:]
            t -= t[0]
            if ~np.isnan(t).all():
                ax[i].plot(
                    t,
                    d * 100 if i else d,
                    label=ch,
                    alpha=0.8,
                    # more color-blind-friendly g, b, and r
                    c={"G": "#009E73", "Iso": "#0072B2", "R": "#D55E00"}.get(
                        ch, f"C{i}"
                    ),
                )
        if i == 0:
            ax[i].legend()
        ax[i].set_title(
            (
                "Raw",
                r"$\Delta$F/F ('dff')",
                r"$\Delta$F/F + motion-correction ('dff_mc')",
            )[i]
        )
        ax[i].set_ylabel(("F [a.u.]", r"$\Delta$F/F [%]", r"$\Delta$F/F [%]")[i])
    tmin, tmax = np.nanmin(t), np.nanmax(t)
    ax[i].set_xlim(tmin - (tmax - tmin) / 100, tmax + (tmax - tmin) / 100)
    plt.suptitle(f"Method: {method},  ROI: {fiber}", y=1)
    plt.xlabel("Time [" + trace.unit + "]")
    plt.tight_layout(pad=0.2)
    fig_path.mkdir(parents=True, exist_ok=True)
    fig_file = fig_path / f"ROI{fiber}_{method}.png"
    plt.savefig(fig_file, dpi=200)
    plt.close()
    return fig_file


def plot_dff(
    df_fip_pp: pd.DataFrame,
    fiber: str,
    channels: list[str],
    method: str,
    fig_path: Path,
    n_frame_to_cut: int = 100,
    zoom_duration: float | None = 60.0,
) -> None:
    """Plot raw and dF/F photometry traces for multiple channels with optional zoomed insets.

    Creates a multi-panel plot showing raw fluorescence signals and baseline-corrected
    dF/F traces for each channel. When zoom_duration is specified, includes three
    zoomed inset plots showing detailed views of the beginning, middle, and end of
    the recording session.

    Parameters
    ----------
    df_fip_pp : pd.DataFrame
        Preprocessed fiber photometry dataframe containing columns:
        - 'channel': Channel names (e.g., 'G', 'R', 'Iso')
        - 'fiber_number': Fiber/ROI identifier
        - 'preprocess': Preprocessing method name
        - 'time_fip': Timestamps in seconds
        - 'signal': Raw fluorescence signal
        - 'F0': Fitted baseline signal
        - 'dFF': Baseline-corrected dF/F signal
    fiber : str
        Fiber/ROI identifier to plot (should match 'fiber_number' in dataframe).
    channels : list[str]
        List of channel names to include in the plot (e.g., ['G', 'R', 'Iso']).
        Channels will be plotted in sorted order.
    method : str
        Preprocessing method name (should match 'preprocess' in dataframe).
        Used for filtering data and in plot title.
    fig_path : Path
        Directory path where the generated plot will be saved.
        Directory will be created if it doesn't exist.
    n_frame_to_cut : int, optional
        Number of frames to exclude from the beginning when setting y-axis limits
        for raw signal plots. Helps avoid artifacts from recording start.
        Default is 100.
    zoom_duration : float or None, optional
        Duration in seconds for each zoomed inset window. If None or 0,
        creates a simpler 2-row layout without insets. If positive, creates
        a 3-row layout with inset zoom plots. Default is 60.0.

    Returns
    -------
    None
        Saves the plot as a PNG file to the specified path and closes the figure.

    Notes
    -----
    The function creates different layouts based on zoom_duration:

    - If zoom_duration is None or 0: 2 rows per channel (raw + dF/F)
    - If zoom_duration > 0: 3 rows per channel (raw + insets + dF/F)

    Inset windows show:
    - First: Beginning of recording (0 to zoom_duration seconds)
    - Middle: Center portion of recording
    - Last: End of recording (last zoom_duration seconds)

    Color coding follows a consistent scheme:
    - Green ('G'): #009E73
    - Blue ('Iso'): #0072B2
    - Red ('R'): #D55E00
    - Baseline F0: #F0E442 (yellow)

    The saved filename follows the pattern: 'ROI{fiber}_dff-{method}.png'

    Examples
    --------
    >>> # Simple 2-row plot without insets
    >>> plot_dff(df, fiber='1', channels=['G', 'R'], method='poly',
    ...          fig_path='/results/', zoom_duration=None)

    >>> # 3-row plot with 30-second zoom windows
    >>> plot_dff(df, fiber='2', channels=['G', 'Iso', 'R'], method='exp',
    ...          fig_path='/results/', zoom_duration=30.0)
    """
    # Check if we should show insets
    show_insets = zoom_duration is not None and zoom_duration > 0
    rows_per_channel = 3 if show_insets else 2

    # Create figure and axes
    if show_insets:
        # 3-row layout with spacing between channels
        fig = plt.figure(figsize=(12, rows_per_channel * len(channels) + 0.5))
        height_ratios = []
        for c in range(len(channels)):
            height_ratios.extend([1, 1, 1])
            if c < len(channels) - 1:
                height_ratios.append(0.15)

        gs = GridSpec(
            len(height_ratios), 1, figure=fig, height_ratios=height_ratios, hspace=0.1
        )
        ax = []
        subplot_idx = 0
        for c in range(len(channels)):
            for i in range(3):
                ax.append(fig.add_subplot(gs[subplot_idx]))
                subplot_idx += 1
            if c < len(channels) - 1:
                subplot_idx += 1
    else:
        # 2-row layout
        fig, ax = plt.subplots(
            2 * len(channels), 1, figsize=(12, 2 * len(channels)), sharex=True
        )

    colors = {"G": "#009E73", "Iso": "#0072B2", "R": "#D55E00"}

    for c, ch in enumerate(sorted(channels)):
        df = df_fip_pp[
            (df_fip_pp.channel == ch)
            & (df_fip_pp.fiber_number == fiber)
            & (df_fip_pp.preprocess == method)
        ]
        t = df.time_fip.values
        t -= t[0]

        if np.isnan(t).all():
            continue

        color = colors.get(ch, f"C{c}")

        # Calculate row indices based on layout
        raw_idx = 3 * c if show_insets else 2 * c
        dff_idx = 3 * c + 2 if show_insets else 2 * c + 1
        inset_idx = 3 * c + 1 if show_insets else None

        # Plot raw signal (always first row of each channel)
        ax[raw_idx].plot(t, df.signal, label=f"raw {ch}", c=color)
        ax[raw_idx].plot(t, df.F0, label=r"fitted F$_0$", c="#F0E442")
        mi, ma = df.signal[n_frame_to_cut:].min(), df.signal[n_frame_to_cut:].max()
        ax[raw_idx].set_ylim(mi - 0.06 * (ma - mi), ma + 0.14 * (ma - mi))
        ax[raw_idx].legend(
            loc=(0.805, 0.77), ncol=2, borderpad=0.05
        ).get_frame().set_linewidth(0.0)
        ax[raw_idx].set_ylabel("F [a.u.]")

        # Plot dF/F signal (always last row of each channel)
        ax[dff_idx].plot(
            t, df.dFF * 100, label=f"$\\Delta$F/F {ch}", c=color, zorder=-1
        )
        ax[dff_idx].axhline(0, c="k", ls="--")
        ax[dff_idx].legend(
            loc=(0.805, 0.77), ncol=1, borderpad=0.05
        ).get_frame().set_linewidth(0.0)
        ax[dff_idx].set_ylabel(r"$\Delta$F/F [%]", y=1 if show_insets else 0.5)

        # Share x-axis between raw and dF/F
        if show_insets:
            ax[dff_idx].sharex(ax[raw_idx])

        # Add insets if requested
        if show_insets:
            t_total = t[-1] - t[0]
            zoom_windows = [
                (t[0], t[0] + zoom_duration),
                (
                    t[0] + (t_total - zoom_duration) / 2,
                    t[0] + (t_total + zoom_duration) / 2,
                ),
                (max(t[-1] - zoom_duration, t[0]), t[-1]),
            ]

            ax[inset_idx].axis("off")

            for j, (start_time, end_time) in enumerate(zoom_windows):
                inset_ax = inset_axes(
                    ax[inset_idx],
                    width="100%",
                    height="100%",
                    loc="center",
                    bbox_to_anchor=([0.01, 0.34, 0.67][j], 0.0, 0.32, 0.8),
                    bbox_transform=ax[inset_idx].transAxes,
                )

                mask = (t >= start_time) & (t <= end_time)
                if np.any(mask):
                    t_zoom, dff_zoom = t[mask], df.dFF.values[mask] * 100

                    inset_ax.plot(t_zoom, dff_zoom, c=color, linewidth=1.5)
                    inset_ax.axhline(0, c="k", ls="--")
                    inset_ax.grid(True, alpha=0.8)
                    inset_ax.set_xlim(start_time, end_time)

                    if len(dff_zoom) > 0:
                        y_margin = 0.1 * (dff_zoom.max() - dff_zoom.min())
                        try:
                            inset_ax.set_ylim(
                                np.nanmin(dff_zoom) - y_margin,
                                np.nanmax(dff_zoom) + y_margin,
                            )
                        except ValueError:
                            pass

                    inset_ax.set_title(
                        f"{['First', 'Middle', 'Last'][j]} {zoom_duration:.0f}s",
                        fontsize=10,
                        y=0.94,
                    )
                    inset_ax.set_xticks([])
                    inset_ax.set_yticks([])

                    mark_inset(
                        ax[dff_idx],
                        inset_ax,
                        loc1=1,
                        loc2=3,
                        fc="none",
                        ec="#333333",
                        alpha=0.8,
                        linestyle="--",
                        linewidth=1,
                    )

    # Set x-limits and labels
    tmin, tmax = np.nanmin(t), np.nanmax(t)
    margin = (tmax - tmin) / 100

    for c in range(len(channels)):
        raw_idx = 3 * c if show_insets else 2 * c
        dff_idx = 3 * c + 2 if show_insets else 2 * c + 1

        ax[raw_idx].set_xlim(tmin - margin, tmax + margin)
        ax[dff_idx].set_xlim(tmin - margin, tmax + margin)
        plt.setp(ax[raw_idx].get_xticklabels(), visible=False)

        # Show x-labels only on last channel's dF/F row
        if c < len(channels) - 1:
            plt.setp(ax[dff_idx].get_xticklabels(), visible=False)
        else:
            ax[dff_idx].set_xlabel("Time [s]")

    # Layout adjustments
    if show_insets:
        plt.suptitle(
            f"$\\bf{{\Delta F/F_0}}$  Method: {method},  ROI: {fiber}", y=0.995
        )
        layout_params = {1: (0.935, 0.13), 2: (0.96, 0.07), 3: (0.97, 0.05)}
        top, bottom = layout_params.get(
            len(channels), (0.98, 0.04)
        )  # Default for 4+ channels
        plt.subplots_adjust(hspace=0.3, top=top, bottom=bottom, left=0.06, right=0.995)
    else:
        plt.suptitle(
            f"$\\bf{{\Delta F/F_0}}$  Method: {method},  ROI: {fiber}", y=0.985
        )
        plt.tight_layout(pad=0.2, h_pad=0)

    fig_path.mkdir(parents=True, exist_ok=True)
    fig_file = fig_path / f"ROI{fiber}_dff-{method}.png"
    plt.savefig(fig_file, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def plot_motion_correction(
    df_fip_pp: pd.DataFrame,
    fiber: str,
    channels: list[str],
    method: str,
    fig_path: Path,
    coeffs: list[dict],
    intercepts: list[dict],
    weights: list[dict],
    cutoff_freq_motion: float,
    cutoff_freq_noise: float,
    fs: float = 20,
) -> None:
    """Plot dF/F and motion-corrected dF/F photometry traces for multiple channels.

    Parameters
    ----------
    df_fip_pp : pd.DataFrame
        The dataframe with the preprocessed FIP data containing F, dF/F and F0 traces.
    fiber : str
        The name of the fiber for which the signals should be plotted.
    channels : list of str
        A list of channel names to be plotted (e.g., ['G', 'R', 'Iso']).
    method : str
        The name of the preprocessing method used ("poly", "exp", or "bright").
    fig_path : Path
        The path where the generated plot will be saved.
    coeffs : dict of list of dict
        The regression coefficients for each method/fiber/channel combination.
    intercepts : dict of list of dict
        The regression intercepts for each method/fiber/channel combination.
    weights : dict of list of dict
        The regression weights for each method/fiber/channel combination.
    cutoff_freq_motion : float
        Cutoff frequency of the lowpass Butterworth filter that's only
        applied for estimating the regression coefficient, in Hz.
    cutoff_freq_noise : float
        Cutoff frequency of the lowpass Butterworth filter
        that's applied to filter out noise, in Hz.
    fs : float, optional
        Sampling rate of the signal, in Hz. Defaults to 20.

    Returns
    -------
    None
        The function saves the plot to the specified fig_path.
    """
    cut = cutoff_freq_noise is not None and cutoff_freq_noise < fs / 2
    # more color-blind-friendly g, b, and r
    colors = {"G": "#009E73", "Iso": "#0072B2", "R": "#D55E00", "filtered": "#F0E442"}
    rows = 3 * len(channels) - 3
    fig = plt.figure(figsize=(15, rows))
    gs = GridSpec(rows, 3, width_ratios=[11, 1, 3.4])

    def plot_psd(ax, data, color, cut=False):
        """Helper function to create Power Spectral Density plots.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the PSD.
        data : array-like
            The data to compute the PSD for.
        color : str
            The color to use for the plot.
        cut : bool, optional
            Whether to cut the frequency range. Defaults to False.

        Returns
        -------
        matplotlib.axes.Axes
            The axis with the PSD plot.
        """
        psd = np.array(welch(data * 100, nperseg=1024))[:, 1:-1]
        if cut:
            psd = psd[:, psd[0] < min(0.5, 1.25 * cutoff_freq_noise / fs)]
        ax.loglog(psd[0] * fs, psd[1], c=color)
        return ax

    left_axes = []
    center_axes = []
    right_axes = []
    df_iso = df_fip_pp[
        (df_fip_pp.channel == "Iso")
        & (df_fip_pp.fiber_number == fiber)
        & (df_fip_pp.preprocess == method)
    ]
    t = df_iso.time_fip.values
    t -= t[0]
    for c, ch in enumerate(sorted([ch for ch in channels if ch != "Iso"])):
        df = df_fip_pp[
            (df_fip_pp.channel == ch)
            & (df_fip_pp.fiber_number == fiber)
            & (df_fip_pp.preprocess == method)
        ]
        color = colors.get(ch, f"C{c}")
        coef = coeffs[method][int(fiber)][ch]
        intercept = intercepts[method][int(fiber)][ch]
        weight = weights[method][int(fiber)][ch]
        # Create subplots in the left and center column (sharing x-axis)
        for i in range(3):
            ax = fig.add_subplot(
                gs[3 * c + i, 0], sharex=(None if c + i == 0 else left_axes[0])
            )
            ax2 = fig.add_subplot(
                gs[3 * c + i, 1], sharex=(None if c + i == 0 else center_axes[0])
            )
            if i < 2:
                if cut:
                    sos = butter(N=2, Wn=cutoff_freq_noise, fs=fs, output="sos")
                    noise_filt = lambda x: sosfiltfilt(sos, x)
                else:
                    noise_filt = lambda x: x
                ax.plot(
                    t,
                    (noise_filt(df["dFF"]) if i == 0 else df["filtered"]) * 100,
                    c=color,
                    label=(("", "low-passed ")[i] + ch),
                )
                plot_psd(
                    ax2, df["dFF"] if i == 0 else df["filtered"], color, i == 1 and cut
                )
                if i == 0:
                    ax.plot(
                        t,
                        (intercept + noise_filt(df_iso["dFF"]) * coef) * 100,
                        c=colors["Iso"],
                        label="regressed Iso",
                        alpha=0.8,
                    )
                    plot_psd(ax2.twinx(), df_iso["dFF"], colors["Iso"]).tick_params(
                        axis="y", which="both", colors=colors["Iso"]
                    )
                else:
                    ax2.axvline(cutoff_freq_motion, c="k", ls="--")
                    plot_psd(
                        ax2.twinx(), df_iso["filtered"], colors["filtered"], cut
                    ).tick_params(axis="y", which="both", colors=colors["filtered"])
                ax.plot(
                    t,
                    (intercept + df_iso["filtered"] * coef) * 100,
                    c=colors["filtered"],
                    label="low-passed Iso (scaled)",
                )
            else:
                ax.plot(
                    t, df["motion_corrected"] * 100, c=color, label=f"corrected {ch}"
                )
                plot_psd(ax2, df["motion_corrected"], color, cut)
                if cut:
                    ax2.axvline(cutoff_freq_noise, c="k", ls="--")
            ax.legend(
                ncol=3, loc=(0.01, 0.77), borderpad=0.05
            ).get_frame().set_linewidth(0.0)
            ax2.tick_params(axis="y", which="both", colors=color)
            left_axes.append(ax)
            center_axes.append(ax2)

        # Create subplots in the right column, each spanning 3 rows
        ax = fig.add_subplot(
            gs[3 * c : 3 * c + 3, 2], sharex=(None if c == 0 else right_axes[0])
        )
        sc = ax.scatter(
            df_iso["filtered"] * 100,
            df["filtered"] * 100,
            s=0.02 + 0.08 * weight,
            c=weight,
            label="low-passed",
            alpha=0.5,
        )
        plt.colorbar(sc, fraction=0.05, pad=0.03).set_label("IRLS weight")
        x, y = np.array(ax.get_xlim()), ax.get_ylim()
        ax.plot(x, intercept * 100 + coef * x, c="k", label="regression")
        ax.set_ylim(y)
        ax.legend(
            loc="lower right", markerscale=12, borderpad=0.05
        ).get_frame().set_linewidth(0.0)
        ax.set_ylabel(ch, color=color, labelpad=-3)
        ax.set_title(f"Regression coeff.: {coef:.4f}", fontsize=12, y=0.9)
        right_axes.append(ax)

    # Hide x-tick labels for all but the bottom subplots
    for ax in left_axes[:-1] + center_axes[:-1] + right_axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    tmin, tmax = np.nanmin(t), np.nanmax(t)
    left_axes[-1].set_xlim(tmin - (tmax - tmin) / 100, tmax + (tmax - tmin) / 100)
    left_axes[-1].set_xlabel("Time [s]")
    left_axes[rows // 2].set_ylabel(
        "$\Delta$F/F [%]", y=(1.1, 0.5)[rows % 2], labelpad=10
    )
    center_axes[-1].set_xlabel("Frequency [Hz]")
    center_axes[rows // 2].set_ylabel("PSD", y=(1.1, 0.5)[rows % 2])
    right_axes[-1].set_xlabel("Iso", color=colors["Iso"])
    plt.suptitle(
        f"$\\bf{{Motion\;correction}}$   Methods: {method} & iso-IRLS,  ROI: {fiber}",
        y=0.99,
    )
    plt.tight_layout(pad=0.2, h_pad=0, w_pad=0)
    for ax in center_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 - 0.02, pos.y0, pos.width + 0.015, pos.height])

    fig_path.mkdir(parents=True, exist_ok=True)
    fig_file = fig_path / f"ROI{fiber}_dff-{method}_mc-iso-IRLS.png"
    plt.savefig(fig_file, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def create_metric(fiber, method, reference, value, motion=False):
    """Create a QC metric for baseline or motion correction.

    Parameters
    ----------
    fiber : str
        The fiber/ROI identifier.
    method : str
        The preprocessing method used.
    reference : str
        Path to the reference image for this metric.
    value : float | dict
        Metric value.
    motion : bool, optional
        Whether this is a motion correction metric. Defaults to False.

    Returns
    -------
    QCMetric
        The created quality control metric.
    """
    baselines = {
        "poly": "$$a t^4 + b t^3 + c t^2 + d t + e$$",
        "exp": "$$a \exp(-b t) + c \exp(-d t)$$",
        "tri-exp": "$$a \exp(-b t) + c \exp(-d t) + e \exp(-f t) + g$$",
        "bright": (
            "$$b_{inf} \cdot (1 + b_{slow}\exp(-t/t_{slow}) + b_{fast}\exp(-t/t_{fast}) + "
            "b_{rapid}\exp(-t/t_{rapid})) \cdot (1 - b_{bright}\exp(-t/t_{bright}))$$"
        ),
    }
    return QCMetric(
        name=f"{'Motion' if motion else 'Baseline'} correction of ROI {fiber} using method '{method}'",
        reference=reference,
        status_history=[
            QCStatus(
                evaluator=(
                    "Automatic" if (motion and value > 10) else "Pending review"
                ),
                timestamp=dt.now(),
                status=Status.FAIL if (motion and value > 10) else Status.PENDING,
            )
        ],
        value=value,
        description=(
            "Maximum regression coefficient"
            if motion
            else "Baseline $$F_0(t)$$ fit with  " + baselines[method]
        ),
    )


def create_evaluation(method, metrics):
    """Create a QC evaluation for a specific preprocessing method.

    Parameters
    ----------
    method : str
        The preprocessing method being evaluated.
    metrics : list of QCMetric
        The metrics included in this evaluation.

    Returns
    -------
    QCEvaluation
        The created quality control evaluation.
    """
    name = f"Preprocessing using method '{method}'"
    return QCEvaluation(
        name=name,
        modality=Modality.FIB,
        stage=Stage.PROCESSING,
        metrics=metrics,
        allow_failed_metrics=False,
        description=(
            "Review the preprocessing plots to ensure accurate "
            "baseline (dF/F) and motion correction."
        ),
    )


def _process1channel(channel, df_fip, fiber_number, pp_name):
    """Helper function to process a single channel (must be at module level for pickling)."""
    df_fip_iter = df_fip[
        (df_fip["fiber_number"] == fiber_number) & (df_fip["channel"] == channel)
    ].copy()

    NM_values = df_fip_iter["signal"].values
    timestamps = df_fip_iter["time_fip"].values
    NM_preprocessed, NM_fitting_params, NM_fit = chunk_processing(
        NM_values,
        timestamps - timestamps[0],
        method=pp_name,
    )
    params_str = ", ".join(f"{v:.5g}" for v in NM_fitting_params.values())
    logging.info(
        f"Fitted parameters for {channel:>3}{fiber_number} "
        f"using method '{pp_name}':  {params_str}"
    )
    df_fip_iter.loc[:, "dFF"] = NM_preprocessed
    df_fip_iter.loc[:, "preprocess"] = pp_name
    df_fip_iter.loc[:, "F0"] = NM_fit

    NM_fitting_params.update(
        {
            "preprocess": pp_name,
            "channel": channel,
            "fiber_number": fiber_number,
        }
    )
    df_pp_params_ses = pd.DataFrame(NM_fitting_params, index=[0])
    return df_fip_iter, df_pp_params_ses


def _process1fiber(
    fiber_number,
    df_fip,
    channels,
    pp_name,
    cutoff_freq_motion,
    cutoff_freq_noise,
    serial,
):
    """Helper function to process a single fiber (must be at module level for pickling).

    Parameters
    ----------
    fiber_number : str
        Fiber/ROI number.
    df_fip : pd.DataFrame
        Raw fiber photometry dataframe.
    channels : np.ndarray
        Array of channel names.
    pp_name : str
        Preprocessing method name.
    cutoff_freq_motion : float
        Cutoff frequency for motion filtering.
    cutoff_freq_noise : float
        Cutoff frequency for noise filtering.
    serial : bool
        Whether to process channels serially.

    Returns
    -------
    tuple
        Contains five elements:
        - df_1fiber : pd.DataFrame
            Dataframe with preprocessed fiber photometry signals.
        - df_pp_params : pd.DataFrame
            Dataframe with the parameters of the preprocessing.
        - coeff : dict
            Dictionary mapping channels to regression coefficients for motion correction.
        - intercept : dict
            Dictionary mapping channels to regression intercepts for motion correction.
        - weight : dict
            Dictionary mapping channels to IRLS weights for motion correction.
    """
    # dF/F - process each channel
    if serial:
        res = [_process1channel(ch, df_fip, fiber_number, pp_name) for ch in channels]
    else:
        res = Parallel(n_jobs=len(channels), backend="threading")(
            delayed(_process1channel)(ch, df_fip, fiber_number, pp_name)
            for ch in channels
        )

    df_1fiber = pd.concat([r[0] for r in res], ignore_index=True)
    df_pp_params = pd.concat([r[1] for r in res])

    # Motion correction
    df_dff_iter = pd.DataFrame(  # convert to #frames x #channels
        np.column_stack(
            [df_1fiber[df_1fiber["channel"] == c]["dFF"].values for c in channels]
        ),
        columns=channels,
    )
    # Run motion correction
    df_mc_iter, df_filt_iter, coeff, intercept, weight = motion_correct(
        df_dff_iter,
        cutoff_freq_motion=cutoff_freq_motion,
        cutoff_freq_noise=cutoff_freq_noise,
    )
    # Convert back to a table with columns channel and signal
    df_1fiber["motion_corrected"] = df_mc_iter.melt(
        var_name="channel", value_name="motion_corrected"
    ).motion_corrected
    df_1fiber["filtered"] = df_filt_iter.melt(
        var_name="channel", value_name="filtered"
    ).filtered
    return df_1fiber, df_pp_params, coeff, intercept, weight


def process_nwb_file(
    nwb_file_path: Path,
    args,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, list]:
    """Process a single NWB file: compute dF/F and motion correction.

    Parameters
    ----------
    nwb_file_path : Path
        Path to the NWB file to process (will be written to).
    args : argparse.Namespace
        Command-line arguments containing processing parameters.

    Returns
    -------
    tuple
        Contains:
        - df_fip_pp : pd.DataFrame
            Preprocessed dataframe with dF/F and motion-corrected traces.
        - df_pp_params : pd.DataFrame
            Fitting parameters for each fiber/channel/method.
        - coeffs : dict
            Regression coefficients by method.
        - intercepts : dict
            Regression intercepts by method.
        - weights : dict
            IRLS weights by method.
        - methods : list
            List of preprocessing methods used.
    """
    # Open NWB file and convert to dataframe
    with NWBZarrIO(path=str(nwb_file_path), mode="r") as io:
        nwb_file = io.read()
        df_fip = nwb_utils.nwb_to_dataframe(nwb_file)

    df_fip_pp = pd.DataFrame()
    df_pp_params = pd.DataFrame()
    coeffs, intercepts, weights = {}, {}, {}
    fiber_numbers = df_fip["fiber_number"].unique()
    channels = df_fip["channel"].unique()
    channels = channels[~pd.isna(channels)]

    for pp_name in args.dff_methods:
        if pp_name not in ["poly", "exp", "tri-exp", "bright"]:
            continue

        if args.serial:
            res = [
                _process1fiber(
                    fib,
                    df_fip,
                    channels,
                    pp_name,
                    args.cutoff_freq_motion,
                    args.cutoff_freq_noise,
                    args.serial,
                )
                for fib in fiber_numbers
            ]
        else:
            res = Parallel(n_jobs=-1)(
                delayed(_process1fiber)(
                    fib,
                    df_fip[df_fip.fiber_number == str(fib)],
                    channels,
                    pp_name,
                    args.cutoff_freq_motion,
                    args.cutoff_freq_noise,
                    args.serial,
                )
                for fib in fiber_numbers
            )

        df_fip_pp = pd.concat([df_fip_pp] + [r[0] for r in res])
        df_pp_params = pd.concat([df_pp_params] + [r[1] for r in res])
        coeffs[pp_name] = [r[2] for r in res]
        intercepts[pp_name] = [r[3] for r in res]
        weights[pp_name] = [r[4] for r in res]

    methods = list(df_fip_pp.preprocess.unique())

    # Write back to NWB
    with NWBZarrIO(path=str(nwb_file_path), mode="r+") as io:
        nwb_file = io.read()

        for method in methods:
            for signal, suffix in (
                ("dFF", f"_dff-{method}"),
                ("motion_corrected", f"_dff-{method}_mc-iso-IRLS"),
            ):
                # format the processed traces as dict for conversion to nwb
                dict_from_df = nwb_utils.split_fip_traces(
                    df_fip_pp[df_fip_pp.preprocess == method], signal=signal
                )
                # and add them to the original nwb
                nwb_file = nwb_utils.attach_dict_fip(nwb_file, dict_from_df, suffix)

        io.write(nwb_file)
        logging.info(
            "Successfully updated the nwb with preprocessed data"
            f" using methods {methods}"
        )

    return df_fip_pp, df_pp_params, coeffs, intercepts, weights, methods


def _plot_both(
    fiber,
    method,
    df_fip_pp,
    channels,
    output_dir,
    coeffs,
    intercepts,
    weights,
    cutoff_freq_motion,
    cutoff_freq_noise,
):
    """Helper function to plot both dff and motion correction (must be at module level for pickling)."""
    output_dir = Path(output_dir)
    plot_dff(
        df_fip_pp,
        fiber,
        channels,
        method,
        output_dir / "dff-qc",
    )
    plot_motion_correction(
        df_fip_pp,
        fiber,
        channels,
        method,
        output_dir / "dff-qc",
        coeffs,
        intercepts,
        weights,
        cutoff_freq_motion,
        cutoff_freq_noise,
    )


def _params_as_dict(fiber, method, df_pp_params):
    """Helper function to convert parameters to dict (must be at module level for pickling)."""
    df = df_pp_params[
        (df_pp_params["fiber_number"] == str(fiber))
        & (df_pp_params["preprocess"] == method)
    ][
        ["channel"]
        + list(range({"poly": 5, "exp": 4, "tri-exp": 7, "bright": 9}[method]))
    ]
    param_names = {
        "poly": [*"abcde"],
        "exp": [*"abcd"],
        "tri-exp": [*"abcdefg"],
        "bright": [
            "b_inf",
            "b_slow",
            "b_fast",
            "b_rapid",
            "b_bright",
            "t_slow",
            "t_fast",
            "t_rapid",
            "t_bright",
        ],
    }
    df.columns = ["channel"] + param_names[method]
    return df.to_dict("list")


def generate_qc_plots(
    df_fip_pp: pd.DataFrame,
    df_pp_params: pd.DataFrame,
    coeffs: dict,
    intercepts: dict,
    weights: dict,
    methods: list,
    args,
    output_dir: Path,
) -> QualityControl:
    """Generate QC plots and return QualityControl object.

    Parameters
    ----------
    df_fip_pp : pd.DataFrame
        Preprocessed fiber photometry dataframe.
    df_pp_params : pd.DataFrame
        Dataframe with fitting parameters.
    coeffs : dict
        Regression coefficients by method.
    intercepts : dict
        Regression intercepts by method.
    weights : dict
        IRLS weights by method.
    methods : list
        List of preprocessing methods used.
    args : argparse.Namespace
        Command-line arguments.
    output_dir : Path
        Output directory for QC plots.

    Returns
    -------
    QualityControl
        Quality control object with evaluations.
    """
    channels = df_fip_pp["channel"].unique()
    fibers = df_fip_pp["fiber_number"].unique()

    # Prepare arguments for parallel plotting
    plot_args = [
        (
            fiber,
            method,
            df_fip_pp[
                (df_fip_pp.fiber_number == fiber) & (df_fip_pp.preprocess == method)
            ],
            channels,
            output_dir,
            coeffs,
            intercepts,
            weights,
            args.cutoff_freq_motion,
            args.cutoff_freq_noise,
        )
        for fiber, method in itertools.product(fibers, methods)
    ]

    if args.serial:
        for args_tuple in plot_args:
            _plot_both(*args_tuple)
    else:
        Parallel(n_jobs=-1)(
            delayed(_plot_both)(*args_tuple) for args_tuple in plot_args
        )

    evaluations = []
    for method in methods:
        metrics = []
        for fiber in fibers:
            metrics.append(
                create_metric(
                    fiber,
                    method,
                    f"dff-qc/ROI{fiber}_dff-{method}.png",
                    _params_as_dict(fiber, method, df_pp_params),
                )
            )
            metrics.append(
                create_metric(
                    fiber,
                    method,
                    f"dff-qc/ROI{fiber}_dff-{method}_mc-iso-IRLS.png",
                    max(v for k, v in coeffs[method][int(fiber)].items() if k != "Iso"),
                    True,
                )
            )
        evaluations.append(create_evaluation(method, metrics))

    # Create QC object and save
    qc = QualityControl(evaluations=evaluations)
    qc.write_standard_file(output_directory=output_dir / "dff-qc")
    return qc


def main():
    start_time = dt.now()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_pattern",
        type=str,
        default=r"/data/fib_raw_nwb/nwb.zarr",
        help="Source pattern to find nwb input files",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="/results/", help="Output directory"
    )
    parser.add_argument(
        "--fiber_path",
        type=str,
        default="/data/fiber_raw_data",
        help="Directory of fiber raw data",
    )
    parser.add_argument(
        "--dff_methods",
        nargs="+",
        default=["poly", "exp", "bright"],
        help=(
            "List of dff methods to run. Available options are:\n"
            "  'poly': Fit with 4th order polynomial using ordinary least squares (OLS)\n"
            "  'exp': Fit with biphasic exponential using OLS\n"
            "  'tri-exp': Fit with triphasic exponential using OLS\n"
            "  'bright': Robust fit with [Bi- or Tri-phasic exponential decay (bleaching)] x "
            "[Increasing saturating exponential (brightening)] using iteratively "
            "reweighted least squares (IRLS)"
        ),
    )
    parser.add_argument(
        "--cutoff_freq_motion",
        type=float,
        default=0.05,
        help=(
            "Cutoff frequency of the lowpass Butterworth filter that's only "
            "applied for estimating the regression coefficient, in Hz."
        ),
    )
    parser.add_argument(
        "--cutoff_freq_noise",
        type=float,
        default=3,
        help=(
            "Cutoff frequency of the lowpass Butterworth filter "
            "that's applied to filter out noise, in Hz."
        ),
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Do not use multiple processes and threads to parallelize fibers and channels.",
    )
    parser.add_argument("--no_qc", action="store_true", help="Skip QC plots.")
    args = parser.parse_args()
    fiber_path = Path(args.fiber_path)
    output_dir = Path(args.output_dir)
    data_desc_fp = next(fiber_path.rglob("data_description.json"))
    with open(data_desc_fp, "r") as j:
        data_name = json.load(j).get("name")
    # Setup logging
    setup_logging_from_metadata(fiber_path)
    logging.info("Begin processing...", extra={"event_type": "stage_start"})
    # Create the destination directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nwb").mkdir(parents=True, exist_ok=True)
    nwb_path = output_dir / "nwb" / (data_name + ".nwb")
    # Find all files matching the source pattern
    source_paths = glob.glob(args.source_pattern)
    if not source_paths:
        logging.warning(
            "No NWB file found! Did you specify the correct source_pattern?"
        )

    # Copy each matching file to the destination directory
    for source_path in source_paths:
        destination_path = nwb_path
        shutil.copytree(source_path, destination_path)

        # Check if fiber photometry data exists
        has_fiber = (fiber_path / "FIP").is_dir() or (fiber_path / "fib").is_dir()

        if has_fiber:
            # Print the path to ensure correctness
            logging.info(f"Processing NWB file: {destination_path}")

            # Process the NWB file
            df_fip_pp, df_pp_params, coeffs, intercepts, weights, methods = (
                process_nwb_file(destination_path, args)
            )

            # Generate QC plots if requested
            if not args.no_qc:
                generate_qc_plots(
                    df_fip_pp,
                    df_pp_params,
                    coeffs,
                    intercepts,
                    weights,
                    methods,
                    args,
                    output_dir,
                )

            process_name = (
                ProcessName.DF_F_ESTIMATION
            )  # append DataProcess to processing.json

        else:
            logging.info("NO Fiber but only Behavior data, preprocessing not needed")
            qc_dir = output_dir / "dff-qc"
            qc_dir.mkdir(parents=True, exist_ok=True)
            qc_file_path = qc_dir / "no_fip_to_qc.txt"
            # Create an empty file
            qc_file_path.write_text(
                "FIP data files are missing. This may be a behavior session."
            )
            process_name = None  # update processing.json w/o appending DataProcess

        write_output_metadata(
            metadata=vars(args),
            json_dir=fiber_path,
            process_name=process_name,
            input_fp=source_path,
            output_fp=output_dir,
            start_date_time=start_time,
        )

    # Iterate over all .json files in the source directory
    for filename in ["subject.json", "procedures.json", "session.json", "rig.json"]:
        src_file = fiber_path / filename
        if src_file.exists():
            dest_file = output_dir / filename
            shutil.copy2(src_file, dest_file)
            logging.info(f"Copied: {src_file} to {dest_file}")
    logging.info("Capsule stage completed", extra={"event_type": "stage_complete"})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(
            "Pipeline stage failed",
            extra={"event_type": "stage_error"}
        )
        logging.exception(e)
        raise