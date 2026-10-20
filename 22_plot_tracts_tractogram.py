import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from dipy.io.streamline import load_tractogram
from fury import actor, window


def load_streamlines(path):
    """Load a .trx/.trk/.tck file into RASMM streamlines."""
    sft = load_tractogram(path, reference="same")   # trx carries its own header
    sft.to_rasmm()
    return sft.streamlines


def plot_tracts(participant, tract_names, afq_TR_path,
                whole_brain_path=None, n_background=20000,
                cmap="tab10", line_width=4, bg_opacity=0.15,
                bg_color=(0.8, 0.8, 0.8), background=(0, 0, 0),
                size=(1200, 1200), save_path=None, interactive=True,
                seed=0):

    scene = window.Scene()
    scene.background(background)

    # ---- Whole-brain tractogram (subsampled for speed) ----
    if whole_brain_path is not None:
        wb = load_streamlines(whole_brain_path)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(wb), size=min(n_background, len(wb)),
                         replace=False)
        wb_sub = [wb[i] for i in idx]
        scene.add(actor.line(wb_sub, colors=bg_color,
                             opacity=bg_opacity, linewidth=1))

    # ---- Individual tracts, one color each ----
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(len(tract_names), 2)))
    color_map = {}
    for tract, color in zip(tract_names, colors):
        f = op.join(afq_TR_path, participant, "bundles",
                    f"{participant}_ses-concat_acq-HCPdir99_desc-{tract}_tractography.trx")
        if not op.exists(f):
            print(f"Missing: {f}")
            continue
        sl = load_streamlines(f)
        rgb = tuple(color[:3])
        color_map[tract] = rgb
        scene.add(actor.line(sl, colors=rgb, opacity=1.0,
                             linewidth=line_width))

    if save_path:
        window.record(scene=scene, out_path=save_path, size=size)
    if interactive:
        window.show(scene, size=size)
    return color_map


# Example
tract_list = list([
'PTR', 
'InferiorLongitudinal',
'InferiorFrontooccipital',
'SuperiorLongitudinal',
'VerticalOccipital',
'Arcuate',
'PosteriorArcuate',
    ])    # or any list of tract names
participant = "sub-EBxGxCCx1986"
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research',
                    'ampb_mt_tractometry_analysis', 'ampb')
afq_TR_path = op.join(bids_path, 'derivatives', 'pyAFQ', 'wmgmi_wb', 'afq3beta-wb_6rounds')
# tract_file = op.join(afq_TR_path, participant, "bundles", f"{participant}_ses-concat_acq-HCPdir99_desc-{tract_name}_tractography.trx")
cmap_used = plot_tracts(
    participant=participant,
    tract_names=tract_list,
    afq_TR_path=afq_TR_path,
    whole_brain_path=op.join(afq_TR_path, participant, 'tractography', f"{participant}_ses-concat_acq-HCPdir99_tractography.trx"),  # your path
    save_path=op.join(bids_path, 'analysis', 'plots', f"tracts_{participant}.png"),
)
print(cmap_used)

