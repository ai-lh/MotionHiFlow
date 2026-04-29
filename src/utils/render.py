# Inspired by
# - https://github.com/anindita127/Complextext2animation/blob/main/src/utils/visualization.py
# - https://github.com/facebookresearch/QuaterNet/blob/main/common/visualization.py

from typing import List, Tuple
import numpy as np
import logging
from src.utils.paramUtil import kit_kinematic_chain, t2m_kinematic_chain, mmm_to_smplh_scaling_factor

mmm_colors = ['black', 'magenta', 'red', 'green', 'blue']

def _divide(N=1):
    ret = 1
    for i in range(2, N):
        if i * i > N:
            break
        if N % i == 0:
            ret = i
    return ret, N // ret


def init_axis(fig, title, radius=1.5, dist=10, total=1, _w=None, index=1):
    if _w is None:
        _h, _w = _divide(total)
    else:
        _h = total // _w
        assert _w * _h == total, f"total={total} is not divisible by _w={_w}"
    ax = fig.add_subplot(_h, _w, index, projection='3d')
    ax.view_init(elev=20., azim=-60)

    fact = 2
    ax.set_xlim3d([-radius / fact, radius / fact])
    ax.set_ylim3d([-radius / fact, radius / fact])
    ax.set_zlim3d([0, radius])

    ax.set_aspect('auto')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_axis_off()

    ax.dist = dist
    ax.grid(b=False)

    ax.set_title(title, loc='center', wrap=True)
    return ax


def plot_floor(ax, minx, maxx, miny, maxy, minz):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Plot a plane XZ
    verts = [
        [minx, miny, minz],
        [minx, maxy, minz],
        [maxx, maxy, minz],
        [maxx, miny, minz]
    ]
    xz_plane = Poly3DCollection([verts], zorder=1)
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 1))
    ax.add_collection3d(xz_plane)

    # Plot a bigger square plane XZ
    radius = max((maxx - minx), (maxy - miny))

    # center +- radius
    minx_all = (maxx + minx) / 2 - radius
    maxx_all = (maxx + minx) / 2 + radius

    miny_all = (maxy + miny) / 2 - radius
    maxy_all = (maxy + miny) / 2 + radius

    verts = [
        [minx_all, miny_all, minz],
        [minx_all, maxy_all, minz],
        [maxx_all, maxy_all, minz],
        [maxx_all, miny_all, minz]
    ]
    xz_plane = Poly3DCollection([verts], zorder=1)
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(xz_plane)
    return ax


def update_camera(ax, root, radius=1.5):
    fact = 2
    ax.set_xlim3d([-radius / fact + root[0], radius / fact + root[0]])
    ax.set_ylim3d([-radius / fact + root[1], radius / fact + root[1]])



def render_animations(jointss: List[np.ndarray], output: str = "notebook", titles: List[str] = [''], title="",
                      fps: float = 20, output_dir=None,
                      kinematic_tree: List[List[int]] | str = "auto",
                      colors: List[str] = mmm_colors,
                      figsize: Tuple[int, int] = (4, 4), is_photo=False, num_actions=3,
                      fontsize: int = 15, _w=None, disable_tqdm=True):
    """ 
    args
    ---
    - jointss (List[np.ndarray[L, js, 3]]): where L is the number of frames, js is the number of joints
    - output (str): output file path or "notebook". default is "notebook"
    - titles (List[str]): title of the plot. default is [""]
    - fps (float): frames per second. default is 20
    - kinematic_tree (List[List[int]] | str): kinematic tree of the joints. default is "mmm", you can also use "humanml3d"
    - colors (List[str]): colors of the joints. default is mmm_colors
    - figsize (Tuple[int]): figure size. default is (4, 4)
    - is_photo (bool): whether to save as photo. default is False
    - fontsize (int): font size. default is 15
    - _w (int): number of columns. default is None
    - disable_tqdm (bool): disable tqdm. default is True
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.patheffects as pe
    plt.rcParams.update({'font.size': fontsize})

    if is_photo == "seq" and len(jointss) == 1:
        # is_photo = False
        _jointss = []
        for i in range(len(jointss[0])):
            if i % num_actions == 0:
                _jointss.append(jointss[0][i:i+1])
        jointss = _jointss
        logging.info(f"jointss is divided into {len(jointss)} parts")


    # Z is gravity here
    x, y, z = 0, 1, 2

    if isinstance(kinematic_tree, str):
        if kinematic_tree.lower() == 'auto':
            if len(jointss) == 0:
                logging.warning("jointss is empty, cannot determine kinematic_tree, set to mmm by default")
                return None
            if jointss[0].shape[1] == 22:
                kinematic_tree = t2m_kinematic_chain
            elif jointss[0].shape[1] == 21:
                kinematic_tree = kit_kinematic_chain
            else:
                raise ValueError(f"Unknown joint number={jointss[0].shape[1]}, cannot determine kinematic_tree, please specify kinematic_tree")
        elif kinematic_tree.lower() in ["mmm", "kit"]:
            kinematic_tree = kit_kinematic_chain
        elif kinematic_tree.lower() in ["humanml3d", "t2m"]:
            kinematic_tree = t2m_kinematic_chain
        else:
            raise ValueError(f"Unknown kinematic_tree={kinematic_tree}")
    # Create a figure and initialize 3d plot
    if _w is None:
        _h, _w = _divide(len(jointss))
        figsize =  (_w * figsize[0], _h * figsize[1])
    else:
        _h = len(jointss) // _w
        assert _w * _h == len(jointss), f"total={len(jointss)} is not divisible by _w={_w}"
        figsize = (_w * figsize[0], _h * figsize[1])
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, horizontalalignment='center', wrap=True, fontsize=fontsize)
    axs = []
    for id, joints in enumerate(jointss):
        # Convert mmm joints for visualization
        # into smpl-h "scale" and axis
        if kinematic_tree is kit_kinematic_chain:
            joints = joints.copy()[..., [2, 0, 1]] * mmm_to_smplh_scaling_factor
        else:
            joints = joints.copy()[..., [2, 0, 1]]

        ax = init_axis(fig, titles[id] if id < len(titles) else "", total=len(jointss), index=id+1, _w=_w)

        # Create spline line
        trajectory = joints[:, 0, [x, y]]
        avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
        if np.isnan(avg_segment_length): # if NaN for single frame
            avg_segment_length = 1e-3
        draw_offset = int(25 / avg_segment_length)
        spline_line, = ax.plot(*trajectory.T, zorder=10, color="white")

        # Create a floor
        minx, miny, _ = joints.min(axis=(0, 1))
        maxx, maxy, _ = joints.max(axis=(0, 1))
        plot_floor(ax, minx, maxx, miny, maxy, 0)

        # Put the character on the floor
        height_offset = np.min(joints[:, :, z])  # Min height
        joints = joints.copy()
        joints[:, :, z] -= height_offset

        # Initialization for redrawing
        lines = []
        initialized = False
        axs.append((ax, joints, trajectory, draw_offset, spline_line, lines, initialized))

    fig.tight_layout()
    frames = max([joints.shape[0] for _, joints, _, _, _, _, _ in axs])
    photo_fps = max(frames // num_actions, 1)
    if is_photo: # set camera
        radius, fact = 1.5, 2
        for ax, joints, _, _, _, _, _ in axs:
            min_x = min([j[0][0] for j in joints])
            min_y = min([j[0][1] for j in joints])
            max_x = max([j[0][0] for j in joints])
            max_y = max([j[0][1] for j in joints])
            ax.set_xlim3d([-radius / fact + min_x, radius / fact + max_x])
            ax.set_ylim3d([-radius / fact + min_y, radius / fact + max_y])

    from tqdm import tqdm
    with tqdm(total=frames, desc="animating..", unit="frame", disable=disable_tqdm) as pbar:
        def update(frame):
            nonlocal axs, pbar
            for id, (ax, joints, trajectory, draw_offset, spline_line, lines, initialized) in enumerate(axs):
                if frame >= joints.shape[0]:
                    continue
                if is_photo and initialized: # 加入轨迹
                    last_skeleton = joints[frame - 1]
                    current_skeleton = joints[frame]
                    for cur_j in range(len(current_skeleton)):
                        ax.plot([last_skeleton[cur_j, x], current_skeleton[cur_j, x]],
                                [last_skeleton[cur_j, y], current_skeleton[cur_j, y]],
                                [last_skeleton[cur_j, z], current_skeleton[cur_j, z]], linewidth=1.0, color="aquamarine", zorder=10)
                if is_photo and frame % photo_fps != 0:
                    continue
                skeleton = joints[frame]

                root = skeleton[0]
                if not is_photo:
                    update_camera(ax, root)

                for index, (chain, color) in enumerate(zip(reversed(kinematic_tree), reversed(colors))):
                    if not initialized or is_photo:
                        lines.append(ax.plot(skeleton[chain, x],
                                            skeleton[chain, y],
                                            skeleton[chain, z], linewidth=3.0, color=color, zorder=20,
                                            path_effects=[pe.SimpleLineShadow(), pe.Normal()]))
                    else:
                        lines[index][0].set_xdata(skeleton[chain, x])
                        lines[index][0].set_ydata(skeleton[chain, y])
                        lines[index][0].set_3d_properties(skeleton[chain, z])

                left = max(frame - draw_offset, 0)
                right = min(frame + draw_offset, trajectory.shape[0])

                spline_line.set_xdata(trajectory[left:right, 0])
                spline_line.set_ydata(trajectory[left:right, 1])
                spline_line.set_3d_properties(np.zeros_like(trajectory[left:right, 0]))
                axs[id] = (ax, joints, trajectory, draw_offset, spline_line, lines, True)
            pbar.update(1)
        
        if output != "notebook":
            import os
            if output_dir != None:
                output = os.path.join(output_dir, output)
            os.makedirs(os.path.dirname(output), exist_ok=True)
        if is_photo:
            for frame in range(frames):
                update(frame)
            if output == "notebook":
                ret = fig
                # from IPython.display import Image
                # ret = Image(fig)
            else:
                fig.savefig(output)
                ret = output
        else:
            anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=False)

            if output == "notebook":
                from IPython.display import HTML
                ret = HTML(anim.to_jshtml())
            else:
                # anim.save(output, writer='ffmpeg', fps=fps)
                _logger = logging.getLogger()
                _logger.setLevel(logging.ERROR)
                anim.save(output, fps=fps)
                _logger.setLevel(logging.INFO) # prevent from ffmpeg warning / info
                ret = output

    plt.close()
    return ret

