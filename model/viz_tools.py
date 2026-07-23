import io
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from ngclearn.utils.io_utils import makedir

def visualize_saccade_trajectory(
    image: np.ndarray,                # shape: (C, H, W) or (H, W) - raw or filtered image
    coords_history: np.ndarray,       # shape: (N_glimpses, 2) in normalized [-1, 1] range
    fovea_radius_px: float = 8.0,     # size of foveal patch in pixels
    periphery_radius_px: float = 24.0,# size of peripheral patch in pixels
    title: str = "MPC Saccadic Trajectory",
    fname: str = "mpc_saccade.jpg"
):
    # Plots the sequential path of an MPC agent's glimpses over an image.
    # The trajectory color fades from cool (start) to hot (end).

    if image.ndim == 3:
        # If RGB/multichannel, take the luminance or first channel for background clarity
        bg_img = np.mean(image, axis=0) if image.shape[0] == 3 else image[0]
    else:
        bg_img = image

    H, W = bg_img.shape
    num_glimpses = coords_history.shape[0]
    ## convert entire history of [-1, 1] coordinates back to absolute pixel coordinates
    ## match exact transformation logic used inside JAX planning function
    y_pixels = ((coords_history[:, 0] + 1.0) / 2.0) * (H - 1)
    x_pixels = ((coords_history[:, 1] + 1.0) / 2.0) * (W - 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ## render underlying structural layout
    ax.imshow(bg_img, cmap='gray', origin='upper')
    ## generate a color gradient across glimpses (e.g., transitioning from blue to red)
    colors = cm.plasma(np.linspace(0, 1, num_glimpses))

    ## draw directional trajectory lines step-by-step
    for i in range(num_glimpses - 1):
        ax.annotate(
            '', 
            xy=(x_pixels[i+1], y_pixels[i+1]), 
            xytext=(x_pixels[i], y_pixels[i]),
            arrowprops=dict(
                arrowstyle="-|>", 
                color=colors[i], 
                lw=2.0, 
                mutation_scale=15,
                alpha=0.8
            )
        )

    ## overlay foveal and peripheral boundary circles for each glimpse location
    for i in range(num_glimpses):
        # Draw Glimpse Node Center
        ax.scatter(x_pixels[i], y_pixels[i], color=colors[i], s=40, zorder=5, edgecolors='w')
        
        # Number the sequence explicitly to easily track order
        ax.text(x_pixels[i] + 1.5, y_pixels[i] - 1.5, str(i+1), color='white', 
                fontsize=9, weight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=1))

        if fovea_radius_px > 0.:
            ## Draw Foveal Focus Ring (Solid)
            fovea_circle = plt.Circle(
                (x_pixels[i], y_pixels[i]), fovea_radius_px, 
                color=colors[i], fill=False, lw=1.2, ls='-', alpha=0.6
            )
            ax.add_patch(fovea_circle)
        if periphery_radius_px > 0.:
            ## Draw Peripheral Integration Ring (Dashed)
            periphery_circle = plt.Circle(
                (x_pixels[i], y_pixels[i]), periphery_radius_px, 
                color=colors[i], fill=False, lw=1.0, ls='--', alpha=0.3
            )
            ax.add_patch(periphery_circle)

    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0) # Maintain 'upper' image coordinate space matrix alignment
    ax.set_title(title, fontsize=14, pad=10)
    ax.axis('off')
    ## add a colorbar to indicate time/sequence dimension
    sm = cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=1, vmax=num_glimpses))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label('Glimpse Sequence Timeline (Start -> End)', fontsize=10)

    plt.tight_layout()
    plt.savefig(fname)

def plot_diagnostic_panel(
    image: np.ndarray,                # shape: (H, W) or (C, H, W) original/filtered image
    bu_map: np.ndarray,               # bottom-up salience map from model (H, W)
    final_score_map: np.ndarray,      # bounded final score map right before argmax/sampling (H, W)
    coords_history: np.ndarray,       # shape (N, 2) normalized [-1, 1] coordinates
    title: str = "MPC Active Inference Diagnostics", 
    fname: str = "mpc_diagnostics.jpg"
):
    ## renders a composite analytical layout tracking the trajectory,
    ## structural bottom-up energy, and final decision landscape

    ## standardize image channel layouts
    bg_img = np.mean(image, axis=0) if image.ndim == 3 else image
    H, W = bg_img.shape
    num_glimpses = coords_history.shape[0]
    ## map normalized coordinates back to absolute pixel space
    y_pixels = ((coords_history[:, 0] + 1.0) / 2.0) * (H - 1)
    x_pixels = ((coords_history[:, 1] + 1.0) / 2.0) * (W - 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    colors = cm.plasma(np.linspace(0, 1, num_glimpses))

    ##########################################################
    ## PANEL 1: Fading Trajectory + Multi-Scale Glimpse Rings
    ax0 = axes[0]
    ax0.imshow(bg_img, cmap='gray', origin='upper')
    ax0.set_title("1. Dynamic Trajectory Path", fontsize=12, weight='bold')

    ## draw arrow steps
    for i in range(num_glimpses - 1):
        ax0.annotate('', xy=(x_pixels[i+1], y_pixels[i+1]), xytext=(x_pixels[i], y_pixels[i]),
                     arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=2.0, mutation_scale=12, alpha=0.7))

    ## draw scale-specific patch boundary rings (radii = width / 2)
    for i in range(num_glimpses):
        ax0.scatter(x_pixels[i], y_pixels[i], color=colors[i], s=30, zorder=5, edgecolors='w')
        ax0.text(x_pixels[i]+1.5, y_pixels[i]-1.5, str(i+1), color='white', fontsize=8,
                 weight='bold', bbox=dict(facecolor='black', alpha=0.6, pad=0.8))
        ## add 3 concentric structural streams
        ax0.add_patch(plt.Circle((x_pixels[i], y_pixels[i]), 12, color=colors[i], fill=False, lw=1.2, ls='-'))  # Foveal
        ax0.add_patch(plt.Circle((x_pixels[i], y_pixels[i]), 24, color=colors[i], fill=False, lw=0.9, ls='--')) # Parafoveal
        ax0.add_patch(plt.Circle((x_pixels[i], y_pixels[i]), 36, color=colors[i], fill=False, lw=0.7, ls=':'))  # Peripheral

    ##########################################################
    ## PANEL 2: Bottom-Up Salience Matrix
    ax1 = axes[1]
    im1 = ax1.imshow(bu_map, cmap='viridis', origin='upper')
    ax1.set_title("2. Bottom-Up Salience ($bu\_map$)", fontsize=12, weight='bold')
    fig.colorbar(im1, ax=ax1, shrink=0.7, label='Energy Intensity')

    ##########################################################
    ## PANEL 3: Final Combined Bounded Score Landscape
    ax2 = axes[2]
    ## replace mask values (-1e10) with NaN or minimum valid values to make plots readable
    display_score = np.array(final_score_map)
    display_score[display_score < -1e5] = np.nan
    im2 = ax2.imshow(display_score, cmap='magma', origin='upper')
    ax2.set_title("3. Bounded Score Map ($final\_score\_map$)", fontsize=12, weight='bold')
    fig.colorbar(im2, ax=ax2, shrink=0.7, label='Attraction Value')
    ## trace trajectory centers lightly over panels 2 & 3 to cross-reference
    for ax in [ax0, ax1, ax2]:
        ax.set_xlim(0, W - 1)
        ax.set_ylim(H - 1, 0)
        ax.axis('off')

    plt.suptitle(title, fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(fname)

def create_saccade_video(
    image,
    bu_map,
    coords_history,
    score_maps_history,
    latent_distances,
    output_path="saccade_evolution.webp",
    frames_dir_path:str = None
):
    """
    Generates a 4-panel diagnostic MP4 video showing saccade evolution and latent shifts.

    Args:
        image: 2D or 3D numpy array of the original image.
        bu_map: 2D numpy array of the static LGN bottom-up salience map.
        coords_history: Numpy array of shape (N, 2) containing normalized (x, y) coordinates.
        score_maps_history: List of N 2D numpy arrays representing final_score_map at each step.
        latent_distances: List or array of length (N-1) containing Distance(z_t, z_{t+1}).
        output_path: Target filename (must end in .mp4).

    """
    frames = []
    num_steps = coords_history.shape[0] #len(trajectory_coords)
    bg_img = np.mean(image, axis=0) if image.ndim == 3 else image
    H, W = bg_img.shape
    ## map normalized coordinates back to absolute pixel space
    trajectory_coords = ((coords_history + 1.0) / 2.0) * (np.array([[W, H]]) - 1.)

    ## set up a 4-panel horizontal canvas to accommodate new chart
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.tight_layout()

    for t in range(num_steps):
        ## clear all axes for the fresh step redraw
        for ax in axes:
            ax.clear()

        ###### Panel 1: Saccade Trajectory ######
        axes[0].imshow(bg_img, cmap="gray" if bg_img.ndim == 2 else None)
        axes[0].set_title(f"Saccade Path (Step {t+1}/{num_steps})")

        y_hist = trajectory_coords[: t + 1, 0]
        x_hist = trajectory_coords[: t + 1, 1]

        if t > 0:
            axes[0].plot(
                x_hist, y_hist, color="cyan", linestyle="-", linewidth=1.5, alpha=0.6
            )
            axes[0].scatter(
                x_hist[:-1], y_hist[:-1], color="blue", s=30, alpha=0.5
            )

        axes[0].scatter(
            x_hist[-1], y_hist[-1], color="red", s=100, edgecolors="white", marker="*"
        )
        axes[0].text(
            x_hist[-1] + 2,
            y_hist[-1] + 2,
            f"{t+1}",
            color="yellow",
            fontsize=12,
            weight="bold",
        )
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        ###### Panel 2: Bottom-Up Salience ######
        axes[1].imshow(bu_map, cmap="hot")
        axes[1].set_title("Static BU Salience Map")
        axes[1].scatter(x_hist[-1], y_hist[-1], color="cyan", s=40, marker="x")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        ###### Panel 3: Dynamic Final Score Map ######
        axes[2].imshow(score_maps_history[t], cmap="viridis")
        axes[2].set_title("Dynamic Final Score Map")
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        ###### Panel 4: Rolling Latent Distance Chart ######
        axes[3].set_title("Representational Shift: D(z_t, z_t+1)")
        axes[3].set_xlabel("Saccade Transition (t -> t+1)")
        axes[3].set_ylabel("Distance")

        ## plot historical bars up to current transition
        if t > 0:
            past_indices = np.arange(1, t + 1)
            current_distances = latent_distances[:t]
            bars = axes[3].bar(past_indices, current_distances, color="magenta", alpha=0.7)
            ## highlight most recent transition bar in a different color
            bars[-1].set_color("yellow")
        ## set static limits so chart scales do not jump around erratically
        axes[3].set_xlim(0.5, num_steps + 0.5)
        if len(latent_distances) > 0:
            axes[3].set_ylim(0, max(latent_distances) * 1.1)

        ##### Frame Extraction #####
        if frames_dir_path is not None:
            plt.savefig(frames_dir_path + f"frame{t}.jpg", format="jpg", bbox_inches="tight", dpi=100)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        frames.append(iio.imread(buf))
        buf.close()

    plt.close(fig)

    #### Compile Video ####
    # if frames_dir_path is not None:
    #     makedir(frames_dir_path) ## ensure this directory exists!
    #     for t in range(len(frames)):
    #         frame_t = frames[t]
    #         plt.imshow(frame_t)
    #         plt.savefig(frames_dir_path + f"frame{t}", format="jpg", bbox_inches="tight", dpi=100)
    iio.imwrite(output_path, frames, fps=2) ## fps=2 means 2 frames per second, matching 500ms viewing pace
    print(f"Diagnostic video saved successfully to {output_path}")

def plot_information_gain_efficiency(
    steps,
    aif_mean,
    aif_std,
    random_mean,
    random_std,
    output_path="population_ig_final.pdf",
    aif_model_tag="Epistemic Saccade Process", 
    title=None
):
    # Generates the population-level information gain plot with a zoomed inset axis
    # to preserve clarity against the massive random baseline variance.

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    ## main Canvas Layout
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ###### main axis plot
    ## active inference model
    ax.plot(
        steps,
        aif_mean,
        color="#1f77b4",
        linewidth=2.5,
        label=aif_model_tag,
        zorder=4,
    )
    ax.fill_between(
        steps,
        aif_mean - aif_std,
        aif_mean + aif_std,
        color="#1f77b4",
        alpha=0.25,
        edgecolor="none",
        zorder=3,
    )

    ## random saccades
    ax.plot(
        steps,
        random_mean,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label="Random Saccades (Control Baseline)",
        zorder=2,
    )
    ax.fill_between(
        steps,
        random_mean - random_std,
        random_mean + random_std,
        color="#d62728",
        alpha=0.12,
        edgecolor="none",
        zorder=1,
    )

    ## main axis styling
    if title is not None:
        ax.set_title(
            title,
            weight="bold",
            pad=14,
        )
    ax.set_xlabel("Saccade Glimpse Step ($t$)")
    ax.set_ylabel("Information Gain [$\\mathit{max}(D) - D_t$]") #"Information Gain [$\\max(D) - D_t$]")
    ax.set_xlim(min(steps), max(steps))
    #ax.grid(True, linestyle=":", alpha=0.4, color="#cccccc", zorder=0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.9, color="purple", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#e0e0e0")

    ## inset axis plot (zoom-in)
    ## places inset in upper-right region, sizing it to a % width and a % height of main axis
    ax_inset = inset_axes(
        ax, width="30%", height="25%", loc="upper left", borderpad=2 ##"upper right"
    )

    ## re-plot just active inference data in inset
    ax_inset.plot(steps, aif_mean, color="#1f77b4", linewidth=2, zorder=4)
    ax_inset.fill_between(
        steps,
        aif_mean - aif_std,
        aif_mean + aif_std,
        color="#1f77b4",
        alpha=0.25,
        edgecolor="none",
        zorder=3,
    )

    ## plot random mean line just as flat reference baseline at bottom of zoom window
    ax_inset.plot(
        steps, random_mean, color="#d62728", linestyle="--", linewidth=1.5, zorder=2
    )
    ## zoom-in: force tight limits
    ax_inset.set_xlim(min(steps), max(steps))
    ax_inset.set_ylim(-0.5, 0.5)  # crop window for normalized metric
    ## inset aesthetics
    ax_inset.set_title("Inset: Active Inference", fontsize=9, weight="bold", pad=4)
    ax_inset.grid(True, linestyle=":", alpha=0.3, color="#cccccc")
    ax_inset.tick_params(axis="both", which="major", labelsize=8)

    #plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Paper figure with inset saved to {output_path}")
    plt.close()

def plot_knn_tri_curves(
    windows,
    random_acc,
    deterministic_acc,
    stochastic_sigmoid_acc,
    random_std=None,
    deterministic_std=None,
    stochastic_sigmoid_std=None,
    output_path="knn_robustness.pdf",
    title=None
):
    """
    Plots K-NN probe accuracy across boundary windows for three distinct policies:
    (Random, Deterministic Epistemic, and Stochastic-Sigmoid Epistemic).

    Args:
        windows: list/array of the tested window limits (e.g., [0.2, 0.4, 0.6, 0.8, 1.0])

        random_acc: accuracies for random glimpsing

        deterministic_acc: accuracies for deterministic epistemic planner

        stochastic_sigmoid_acc: accuracies for stochastic sigmoidal planner

        output_path: output directory to save to
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ## deterministic epistemic policy
    ax.plot(
        windows,
        deterministic_acc,
        color="#1f77b4",
        linestyle="-",
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="Deterministic Saccade Planning",
        zorder=6,
    )
    if deterministic_std is not None:
        ax.fill_between(
            windows,
            deterministic_acc - deterministic_std,
            deterministic_acc + deterministic_std,
            color="#1f77b4",
            alpha=0.25,
            edgecolor="none",
            zorder=5,
        )

    ## stochastic-sigmoid epistemic policy
    ax.plot(
        windows,
        stochastic_sigmoid_acc,
        color="#ff7f0e",
        linestyle="-.",
        linewidth=2.5,
        marker="^",
        markersize=6,
        label="Stochastic Foraging (Two-Phase Sigmoid)",
        zorder=4,
    )
    if stochastic_sigmoid_std is not None:
        ax.fill_between(
            windows,
            stochastic_sigmoid_acc - stochastic_sigmoid_std,
            stochastic_sigmoid_acc + stochastic_sigmoid_std,
            color="#ff7f0e",
            alpha=0.225,
            edgecolor="none",
            zorder=3,
        )

    ## random saccade control baseline
    ax.plot(
        windows,
        random_acc,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        marker="s",
        markersize=5,
        label="Random Glimpsing (Control Baseline)",
        zorder=2,
    )
    if random_std is not None:
        ax.fill_between(
            windows,
            random_acc - random_std,
            random_acc + random_std,
            color="#d62728",
            alpha=0.2,
            edgecolor="none",
            zorder=1,
        )

    # layout mods
    if title is not None:
        ax.set_title(
            title,
            weight="bold",
            pad=15,
        )
    ax.set_xlabel("Saccade Coordinate Boundary Window ($\pm c$)")
    ax.set_ylabel("Downstream $K$-NN Probe Accuracy (%)")
    # format ticks
    ax.set_xticks(windows)
    ax.set_xticklabels([f"$\\pm${w:.1f}" for w in windows])

    #ax.grid(True, linestyle=":", alpha=0.4, color="#cccccc")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.9, color="purple")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ## place legend cleanly where it will not block data lines
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#e0e0e0")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Triple-curve robustness plot saved to {output_path}")
    plt.close()

def plot_knn_data_subset_curves(
    sample_sizes,
    mpc_acc,
    mlp_acc,
    mpc_std=None,
    mlp_std=None,
    output_path="knn_sample_efficiency.pdf",
    use_logxscale=True,
    use_logyscale=False,
    title=None
): ## plots K-NN probe accuracy across sample sizes for an MLP/DNN versus MPC
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ## MLP baseline curves
    ax.plot(
        sample_sizes,
        mlp_acc,
        color="#1f77b4",

        linestyle="-.",
        linewidth=2.0,
        marker="^",
        markersize=6,
        label="MLP",
        zorder=4,
    )
    if mlp_std is not None:
        ax.fill_between(
            sample_sizes,
            mlp_acc - mlp_std,
            mlp_acc + mlp_std,
            color="#1f77b4",
            alpha=0.225,
            edgecolor="none",
            zorder=3,
        )

    ## MPC curves
    ax.plot(
        sample_sizes,
        mpc_acc,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        marker="s",
        markersize=5,
        label="MPC",
        zorder=2,
    )
    if mpc_std is not None:
        ax.fill_between(
            sample_sizes,
            mpc_acc - mpc_std,
            mpc_acc + mpc_std,
            color="#d62728",
            alpha=0.2,
            edgecolor="none",
            zorder=1,
        )

    ## layout mods
    if title is not None:
        ax.set_title(
            title,
            weight="bold",
            pad=15,
        )
    if use_logyscale:
        ax.set_yscale('log')
    ax.set_xlabel("Dataset Size (# Samples)", fontsize=16)
    ax.set_ylabel("$K$-NN Probe Accuracy (%)", fontsize=16)

    if use_logxscale:
        ax.set_xscale('log')
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([f"{s:.0f}" for s in sample_sizes])

    #ax.grid(True, linestyle=":", alpha=0.4, color="#cccccc")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.9, color="purple")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Sample efficiency plot saved to {output_path}")
    plt.close()

def plot_knn_glimpse_curves(
    sample_sizes,
    mpc_acc,
    mpc_std,
    gpc_acc,
    gpc_std,
    output_path="knn_glimpse_generalization.pdf",
    use_logxscale=True,
    use_logyscale=False,
    title=None
):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ## GPC baseline curves
    if gpc_acc is not None:
        ax.plot(
            sample_sizes,
            gpc_acc,
            color="#1f77b4",
            linestyle="-.",
            linewidth=2.0,
            marker="^",
            markersize=6,
            label="GPC",
            zorder=4,
        )
        ax.fill_between(
            sample_sizes,
            gpc_acc - gpc_std,
            gpc_acc + gpc_std,
            color="#1f77b4",
            alpha=0.225,
            edgecolor="none",
            zorder=3,
        )

    ## MPC curves
    ax.plot(
        sample_sizes,
        mpc_acc,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        marker="s",
        markersize=5,
        label="MPC",
        zorder=2,
    )
    if mpc_std is not None:
        ax.fill_between(
            sample_sizes,
            mpc_acc - mpc_std,
            mpc_acc + mpc_std,
            color="#d62728",
            alpha=0.2,
            edgecolor="none",
            zorder=1,
        )

    ## layout mods
    if title is not None:
        ax.set_title(
            title,
            weight="bold",
            pad=15,
        )
    if use_logyscale:
        ax.set_yscale('log')
    ax.set_xlabel("# of Glimpses", fontsize=16)
    ax.set_ylabel("$K$-NN Probe Accuracy (%)", fontsize=16)

    if use_logxscale:
        ax.set_xscale('log')
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([f"{s:.0f}" for s in sample_sizes])

    #ax.grid(True, linestyle=":", alpha=0.4, color="#cccccc")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.9, color="purple")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Glimpse generalization plot saved to {output_path}")
    plt.close()


