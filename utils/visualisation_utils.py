import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import patches
from PIL import Image

cm = matplotlib.cm.get_cmap('tab20')
def_colors = cm.colors
cus_colors = ['k'] + [def_colors[i] for i in range(1, 19)] + ['w']
cmap = ListedColormap(colors=cus_colors, name='agri', N=20)

label_names = [
    "Background",
    "Meadow",
    "Soft winter wheat",
    "Corn",
    "Winter barley",
    "Winter rapeseed",
    "Spring barley",
    "Sunflower",
    "Grapevine",
    "Beet",
    "Winter triticale",
    "Winter durum wheat",
    "Fruits,  vegetables, flowers",
    "Potatoes",
    "Leguminous fodder",
    "Soybeans",
    "Orchard",
    "Mixed cereal",
    "Sorghum",
    "Void label"]


def get_rgb(x, b=0, t_show=6):
    """Gets an observation from a time series and normalises it for visualisation."""
    im = x[b, t_show, [2, 1, 0]].cpu().numpy()
    mx = im.max(axis=(1, 2))
    mi = im.min(axis=(1, 2))
    im = (im - mi[:, None, None]) / (mx - mi)[:, None, None]
    im = im.swapaxes(0, 2).swapaxes(0, 1)
    im = np.clip(im, a_max=1, a_min=0)
    return im


def plot_pano_predictions(pano_predictions, pano_gt, ax, cmap=cmap, batch_element=0, alpha=.5):
    pano_instances = pano_predictions['pano_instance'][batch_element].squeeze().cpu().numpy()
    pano_semantic_preds = pano_predictions['pano_semantic'][batch_element].argmax(dim=0).squeeze().cpu().numpy()
    grount_truth_semantic = pano_gt[batch_element, :, :, -1].cpu().numpy()

    for inst_id in np.unique(pano_instances):
        if inst_id == 0:
            continue  # ignore background
        mask = (pano_instances == inst_id)
        try:
            # Get polygon contour of the instance mask
            c, h = cv2.findContours(mask.astype(int), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

            # Get the ground truth semantic label of the segment
            u, cnt = np.unique(grount_truth_semantic[mask], return_counts=True)
            cl = u if np.isscalar(u) else u[np.argmax(cnt)]
            if cl == 19:  # Not showing predictions for "Void" segments
                continue

            # Get the predicted semantic label of the segment
            cl = pano_semantic_preds[mask].mean()
            color = cmap.colors[int(cl)]
            for co in c[0::2]:
                poly = patches.Polygon(co[:, 0, :], fill=True, alpha=alpha, linewidth=0, color=color)
                ax.add_patch(poly)
                poly = patches.Polygon(co[:, 0, :], fill=False, alpha=.8, linewidth=4, color=color)
                ax.add_patch(poly)
        except ValueError as e:
            print(cv2.findContours(mask.astype(int), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE))


def plot_pano_gt(pano_gt, cmap=cmap, batch_element=0, alpha=.5, plot_void=True):
    ground_truth_instances = pano_gt[batch_element, :, :, 1].cpu().numpy()
    grount_truth_semantic = pano_gt[batch_element, :, :, -1].cpu().numpy()

    for inst_id in np.unique(ground_truth_instances):
        if inst_id == 0:
            continue
        mask = (ground_truth_instances == inst_id)
        try:
            c, h = cv2.findContours(mask.astype(int), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
            u, cnt = np.unique(grount_truth_semantic[mask], return_counts=True)
            cl = u if np.isscalar(u) else u[np.argmax(cnt)]

            if cl == 19 and not plot_void:  # Not showing predictions for Void objects
                continue

            color = cmap.colors[int(cl)]
            for co in c[1::2]:
                poly = patches.Polygon(co[:, 0, :], fill=True, alpha=alpha, linewidth=0, color=color)
                plt.gca().add_patch(poly)
                poly = patches.Polygon(co[:, 0, :], fill=False, alpha=.8, linewidth=4, color=color)
                plt.gca().add_patch(poly)
        except ValueError as e:
            print(e)


def visual(images, gt_masks, fake_predictions, predict, batch_size, writer, epoch, save_path, mode):
    # Visualisation of the semantic and panoptic predictions
    size = 2
    noisy_steps = len(fake_predictions)
    fig, axes = plt.subplots(batch_size, 3 + noisy_steps, figsize=((3 + noisy_steps) * size, batch_size * size))
    t = 2
    alpha = .5

    for b in range(batch_size):
        # Plot S2 background
        im = get_rgb(images, b=b, t_show=t)
        axes[b, 0].imshow(im)
        # axes[b, 2].imshow(im)
        # axes[b, 1].imshow(im)

        ## Plot ground truth instances
        # plot_pano_gt(pano_gt=gt_masks,
        #              axes=axes,
        #              ax=axes[b, 1],
        #              cmap=cmap,
        #              batch_element=b,
        #              alpha=alpha,
        #              plot_void=True)

        # ## Plot predicted instances
        # plot_pano_predictions(pano_predictions=predictions,
        #                       pano_gt=gt_masks,
        #                       ax=axes[b, 2],
        #                       cmap=cmap,
        #                       batch_element=b,
        #                       alpha=alpha)

        # Plot Semantic Segmentation prediction
        axes[b, 1].matshow(gt_masks[b].cpu().numpy(),
                           cmap=cmap,
                           vmin=0,
                           vmax=19)

        for i in range(len(fake_predictions)):
            prediction = fake_predictions[i].argmax(dim=1)
            # breakpoint()
            axes[b, i + 2].matshow(prediction[b].cpu().numpy(),
                                   cmap=cmap,
                                   vmin=0,
                                   vmax=19)
        axes[b, 2 + noisy_steps].matshow(predict.argmax(dim=1)[b].cpu().numpy(),
                                         cmap=cmap,
                                         vmin=0,
                                         vmax=19)

        for i in range(3 + noisy_steps):
            axes[b, i].axis('off')

        axes[0, 0].set_title('Original image')
        axes[0, 1].set_title('Panoptic Annotation')
        for i in range(noisy_steps):
            axes[0, i + 2].set_title(f't={i + 1}')
        axes[0, 2 + noisy_steps].set_title('prediction')

    # plt.imshow()
    val_images_fold = f'{save_path}val_images'
    if not os.path.exists(val_images_fold):
        os.makedirs(val_images_fold)

    image_path = f'{val_images_fold}/{mode}_{epoch}.png'
    plt.savefig(image_path)
    plt.close()


def save_images(predictions, config, index):
    save_path = f"{config.show_dir}/Fold_{config.fold}/predictions/"
    batch_size = predictions.shape[0]
    alpha = .5

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for b in range(batch_size):
        plt.figure(figsize=(128, 128), dpi=1)
        if config.MODE == "semantic":
            # Plot Semantic Segmentation prediction
            prediction = predictions.argmax(dim=1)
            plt.imshow(prediction[b].cpu().numpy(), cmap=cmap, vmin=0, vmax=19)
        else:
            # Plot Panoptic Segmentation prediction
            plot_pano_gt(pano_gt=predictions,
                         cmap=cmap,
                         batch_element=b,
                         alpha=alpha,
                         plot_void=True)
        save(save_path, f"{config.MODE}_{index + b}.png", True, 1, 0)


def save(save_path, file_name, transparent=True, dpi=1, pad_inches=0):
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(f'{save_path}/{file_name}', transparent=transparent, dpi=dpi, pad_inches=pad_inches)
    plt.close()
