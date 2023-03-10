import supervisely as sly
from supervisely.nn.prediction_dto import PredictionSegmentation
from supervisely.sly_logger import logger
import warnings

warnings.filterwarnings("ignore")

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict
from dotenv import load_dotenv
import os
import torch
import gdown
from src.isnet import ISNetDIS
from src.infer_utils import load_image, build_model, predict
from pathlib import Path


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[1])
model_data_path = os.path.join(root_source_path, "model", "model_data.json")
api = sly.Api()


class ISNetModel(sly.nn.inference.SemanticSegmentation):
    def get_models(self):
        model_data = sly.json.load_json_file(model_data_path)
        return model_data

    def download_weights(self, model_dir):
        model_source = self.gui.get_model_source()
        weights_dst_path = os.path.join(model_dir, "weights.pth")
        if model_source == "Pretrained models":
            if not sly.fs.file_exists(weights_dst_path):
                weights_url = "https://drive.google.com/uc?id=1KyMpRjewZdyYfxHPYcd-ZbanIXtin0Sn"
                gdown.download(weights_url, weights_dst_path, use_cookies=False)
        elif model_source == "Custom models":
            custom_link = self.gui.get_custom_link()
            if not sly.fs.file_exists(weights_dst_path):
                self.download(
                    src_path=custom_link,
                    dst_path=weights_dst_path,
                )

    def prepare_hyperparameters(self, model_dir):
        hypar = {}
        hypar["model_path"] = model_dir  # load trained weights from this path
        hypar["restore_model"] = "weights.pth"  # name of the to-be-loaded weights
        hypar["interm_sup"] = False  # indicate if activate intermediate feature supervision
        hypar["model_digit"] = "full"  # indicates "half" or "full" accuracy of float number
        hypar["seed"] = 0
        # cached input spatial resolution, can be configured into different size
        hypar["cache_size"] = [1024, 1024]
        # model input spatial size, usually use the same value hypar["cache_size"],
        # which means we don't further resize the images
        hypar["input_size"] = [1024, 1024]
        # random crop size from the input, it is usually set as smaller than hypar["cache_size"],
        # e.g., [920, 920] for data augmentation
        hypar["crop_size"] = [1024, 1024]
        # define neural network architecture
        hypar["model"] = ISNetDIS()
        return hypar

    def binarize_mask(self, mask):
        mask[mask < 200] = 0
        mask[mask >= 200] = 1
        return mask

    def _create_label(self, dto: PredictionSegmentation):
        geometry = sly.Bitmap(dto.mask)
        obj_class = self.model_meta.get_obj_class(dto.class_name)
        if not dto.mask.any():  # skip empty masks
            logger.debug(f"Mask is empty and will be sklipped")
            return None
        label = sly.Label(geometry, obj_class)
        return [label]

    @property
    def model_meta(self):
        if self._model_meta is None:
            self._model_meta = sly.ProjectMeta(
                [sly.ObjClass("object_mask", sly.Bitmap, [255, 0, 0])]
            )
            self._get_confidence_tag_meta()
        return self._model_meta

    def load_on_device(
        self,
        model_dir,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        # download weights
        self.download_weights(self._model_dir)
        # set inference parameters
        self.hypar = self.prepare_hyperparameters(self._model_dir)
        # set device
        self.device = device
        # build model
        self.model = build_model(self.hypar, device)
        # define class names
        self.class_names = ["object_mask"]
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_info(self):
        info = super().get_info()
        info["task type"] = "salient object segmentation"
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        info["tracking_on_videos_support"] = False
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[sly.nn.PredictionMask]:
        image_tensor, orig_size = load_image(image_path, self.hypar)
        mask = predict(self.model, image_tensor, orig_size, self.hypar, self.device)
        mask = self.binarize_mask(mask)
        return [sly.nn.PredictionMask(class_name="object_mask", mask=mask)]


m = ISNetModel(
    use_gui=True,
    custom_inference_settings=os.path.join(root_source_path, "custom_settings.yaml"),
)

if sly.is_production():
    m.serve()
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, settings={})
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=0)
    print(f"predictions and visualization have been saved: {vis_path}")
