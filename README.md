<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/115161827/227242096-4d4d9481-d6f9-4032-8977-63901361fa19.jpg"/>  

# Serve IS-Net

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#example-apply-is-net-to-image/roi-in-labeling-tool">Example: apply IS-Net to image in labeling tool</a> •
  <a href="#example-custom-inference-settings/roi-in-labeling-tool">Example: custom inference settings</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/serve-isnet)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/serve-isnet)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/serve-isnet.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/serve-isnet.png)](https://supervise.ly)

</div>

# Overview

IS-Net is a salient object segmentation model which was pretrained on DIS5K dataset to perform highly accurate segmentation of objects with complex contours.
Models under a Salient Instance Segmentation task are usually used for separating foreground from background. They predict a mask for the foreground object. These models are class-agnostic, which means they can't predict a class label for an object.

# How To Run

## Pretrained models

**Step 1.** Select pretrained model architecture and press the **Serve** button

<img src=https://user-images.githubusercontent.com/115161827/228220699-dd8a063d-d9c5-4e29-b5b0-0f0c7bdaa141.png> </img>

**Step 2.** Wait for the model to deploy

<img src=https://user-images.githubusercontent.com/115161827/228220710-8fd2b3d1-9e0b-4724-8bb7-abcf2a5bd1c3.png> </img>

## Custom models

Copy model file path from Team Files:

https://user-images.githubusercontent.com/91027877/229389158-73a2653e-a8ff-4ab7-803c-5b0fe55cefe3.MP4

# Example: apply IS-Net to image/ROI in labeling tool

Run **NN Image Labeling** app, connect to IS-Net, and click on "Apply model to image", or if you want to apply IS-Net only to the region within the bounding box, select the bbox and click on "Apply model to ROI":

https://user-images.githubusercontent.com/115161827/228263450-f4a4ee6a-b0d3-465d-a1e8-7c872f41ca39.mp4

If you want to change model specific inference settings while working with the model in image labeling interface, go to **inference** tab in the settings section of **Apps** window, and change the parameters to your liking:

https://user-images.githubusercontent.com/115161827/228303154-0e484ab8-7e98-4ded-8e56-3f31e91631e2.mp4

# Related apps

You can use served model in next Supervisely Applications ⬇️

- [Apply Object Segmentor to Images Project](https://ecosystem.supervise.ly/apps/apply-object-segmentor-to-images-project) - app allows to label images project using served  detection and pose estimation models.
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-object-segmentor-to-images-project" src="https://user-images.githubusercontent.com/115161827/229510088-dfe8413f-ec09-4cca-988e-596aab4dd7d2.jpg" height="70px" margin-bottom="20px"/>
    
- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployed NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>
    
# Acknowledgment

This app is based on the great work `IS-Net` ([github](https://github.com/xuebinqin/DIS?ysclid=lfs48vrw5740792321)). ![GitHub Org's stars](https://img.shields.io/github/stars/xuebinqin/DIS?style=social)
