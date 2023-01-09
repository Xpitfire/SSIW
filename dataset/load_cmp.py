import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class CMPDataset(Dataset):
    num_labels = 13
    
    unique_colors = [
        [  0,   0,   0],
        [  0,   0, 170],
        [  0,   0, 255],
        [  0,  85, 255],
        [  0, 170, 255],
        [  0, 255, 255],
        [ 85, 255, 170],
        [170,   0,   0],
        [170, 255,  85],
        [255,   0,   0],
        [255,  85,   0],
        [255, 170,   0],
        [255, 255,   0]
    ]
    
    id2label = {
        0: "unknown",
        1: "facade",
        2: "molding",
        3: "cornice",
        4: "pillar",
        5: "window",
        6: "door",
        7: "sill",
        8: "blind",
        9: "balcony",
        10: "shop",
        11: "deco",
        12: "background"
    }
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.lbl_paths = []
        self.label2id = {lbl: id for id, lbl in self.id2label.items()}

        # Iterate over the files in the root directory
        for file in sorted(os.listdir(root_dir)):
            # Check if the file is a JPG image
            if file.endswith(".jpg"):
                self.img_paths.append(os.path.join(root_dir, file))
            # Check if the file is an XML file
            elif file.endswith(".png"):
                self.lbl_paths.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        lbl_path = self.lbl_paths[idx]

        img = Image.open(img_path).convert("RGB").copy()
        mask = Image.open(lbl_path).convert('P').copy()

        if self.transform:
            img, mask = self.transform(img), self.transform(mask)

        return img, mask
    
    def parse_segmentation_image(self, segmentation_image):
        """
        Map RGB pixel values in the segmentation image to class ids.

        Args:
            segmentation_image: a WxHx3 numpy array representing the segmentation image
            unique_colors: a list of unique RGB values in the image

        Returns:
            A WxH numpy array where each element is the class id for the corresponding
            pixel in the segmentation image.
        """
        # Create a mapping from RGB values to class ids
        color_to_class = {tuple(color): i for i, color in enumerate(self.unique_colors)}

        # Initialize the output array with the correct number of classes
        output = np.zeros((segmentation_image.shape[0], segmentation_image.shape[1]), dtype=np.uint8)

        # Map the RGB values to class ids
        for i in range(segmentation_image.shape[0]):
            for j in range(segmentation_image.shape[1]):
                output[i, j] = color_to_class[tuple(segmentation_image[i, j])]

        return output


if __name__ == '__main__':
    import numpy as np
    from dataset.load_cmp import CMPDataset
    from datasets import Dataset

    cmp_ds_train = CMPDataset(root_dir='data/train')
    cmp_ds_eval = CMPDataset(root_dir='data/eval')
    cmp_ds_test = CMPDataset(root_dir='data/test')
    id2label = CMPDataset.id2label
    label2id = {v: k for k, v in id2label.items()}
    num_labels = CMPDataset.num_labels

    imgs = []
    lbls = []
    for img, lbl in cmp_ds_train:
        imgs.append(img)
        lbls.append(lbl)
    train_ds = Dataset.from_dict({"pixel_values": imgs, "label": lbls})
    imgs = []
    lbls = []
    for img, lbl in cmp_ds_eval:
        imgs.append(img)
        lbls.append(lbl)
    eval_ds = Dataset.from_dict({"pixel_values": imgs, "label": lbls})
    imgs = []
    lbls = []
    for img, lbl in cmp_ds_test:
        imgs.append(img)
        lbls.append(lbl)
    test_ds = Dataset.from_dict({"pixel_values": imgs, "label": lbls})
    
    train_ds.save_to_disk('data/cmp/train/hf')
    eval_ds.save_to_disk('data/cmp/eval/hf')
    test_ds.save_to_disk('data/cmp/test/hf')
