import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset


class CMPDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = [os.path.join(root_dir, "cmp_b{:04d}.jpg".format(i)) for i in range(1, 1404)]
        self.xml_paths = [os.path.join(root_dir, "cmp_b{:04d}.xml".format(i)) for i in range(1, 1404)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        xml_path = self.xml_paths[idx]

        img = Image.open(img_path).convert("RGB")
        mask, label = self.parse_xml(xml_path, img.size)

        if self.transform:
            img, mask = self.transform(img), self.transform(mask)

        return img, mask, label

    def parse_xml(self, xml_path, img_size):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        mask = Image.new("L", img_size, 0)
        label = 0

        for object_ in root.findall("object"):
            points = object_.find("points")
            x1 = int(float(points.find("x").text) * img_size[0])
            x2 = int(float(points.find("x").tail) * img_size[0])
            y1 = int(float(points.find("y").text) * img_size[1])
            y2 = int(float(points.find("y").tail) * img_size[1])
            label = int(object_.find("label").text)

            mask.paste(label, (x1, y1, x2, y2))

        return mask, label

