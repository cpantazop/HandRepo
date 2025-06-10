# HandRepo
This repository hosts the code and notes developed during a special topic project titled **A study on 2D/3D hand representation for computer vision applications**. This initial study led to the formulation of my thesis titled **Articulated 3D Hand from a Single RGB Image**, which was later formalized into a publication: **Monocular 3D Hand Pose Estimation with Implicit Camera Alignment**. The work focuses on estimating the 3D articulation of the human hand from a single RGB image, without requiring knowledge of camera parameters.

For full details, please see the [paper](xxxx).

## Installation

To install the necessary dependencies, please follow the installation instructions from the [manotorch repository](https://github.com/lixiny/manotorch), as this implementation builds upon it. 

Additionally, for performing quantitative evaluations, we followed the evaluation setup from [Minimal-Hand-pytorch](https://github.com/MengHao666/Minimal-Hand-pytorch), ensuring consistency and direct comparability with prior works. Therefore, to successfully run our evaluation code, it is also necessary to follow the installation instructions provided by the Minimal-Hand-pytorch repository.

The required models for MediaPipe and MANO have to be downloaded separately from the official sources:  
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)  
- [MANO Hand Model](https://mano.is.tue.mpg.de)
Then the following files have to be included in the `models/` directory:
- MANO_LEFT.pkl
- MANO_RIGHT.pkl
- hand_landmarker.task


# A study on 2D/3D hand representation for computer vision applications
The following notes originate from the initial study phase and are preserved here for reference.

## Contents

- [2D hand pose estimation](#2D-hand-pose-estimation)
- [3D hand pose estimation](#3D-hand-pose-estimation)
- [3D hand pose estimation from RGB Image](#3D-hand-pose-estimation-from-RGB-Image)
- [Datasets](#Datasets)
- [Hand parametric models](#Hand-parametric-models)
- [Real-time 2D/3D applications](#Real-time-2D/3D-applications)

**Gentle Introduction to 2d hand pose estimation**: [read this blog](https://towardsdatascience.com/gentle-introduction-to-2d-hand-pose-estimation-approach-explained-4348d6d79b11)

**Paper to begin searching**: [complex hand keypoints detection](https://arxiv.org/pdf/1704.07809)
## 2D hand pose estimation
- **A Comprehensive Study on Deep Learning-Based 3D Hand Pose Estimation Methods**: [read pdf](https://www.mdpi.com/2076-3417/10/19/6850/pdf?version=1601434111)  
  Theocharis Chatzis , Andreas Stergioulas , Dimitrios Konstantinidis , Kosmas Dimitropoulos and Petros Daras
- **Hand Keypoint Detection in Single Images using Multiview Bootstrapping**: [read pdf](https://arxiv.org/pdf/1704.07809.pdf)  
  Tomas Simon, Hanbyul Joo, Iain Matthews, Yaser Sheikh
  
## 3D hand pose estimation
- **Hand Pose Estimation via Latent 2.5D Heatmap Regression**: [read pdf](https://arxiv.org/pdf/1804.09534.pdf)  
  Umar Iqbal, Pavlo Molchanov , Thomas Breuel Juergen Gall , Jan Kautz

## 3D hand pose estimation from RGB Image
- **3D Hand Shape and Pose from Images in the Wild**: [read pdf](https://arxiv.org/pdf/1902.03451.pdf)  
  Adnane Boukhayma, Rodrigo de Bem , Philip H.S. Torr

- **Using a single RGB frame for real time 3D hand pose estimation in the wild**: [read pdf](https://arxiv.org/pdf/1712.03866.pdf)  
  Paschalis Panteleris Iason Oikonomidis Antonis Argyros
  
- **FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images**: [read pdf](https://arxiv.org/pdf/1909.04349.pdf)  
  Christian Zimmermann , Duygu Ceylan , Jimei Yang , Bryan Russell , Max Argus , and Thomas Brox
  
- **GANerated Hands for Real-Time 3D Hand Tracking from Monocular RGB**: [read pdf](https://arxiv.org/pdf/1712.01057.pdf)  
  Franziska Mueller, Florian Bernard, Oleksandr Sotnychenko, Dushyant Mehta, Srinath Sridhar, Dan Casas, Christian Theobalt
    
- **Learning to Estimate 3D Hand Pose from Single RGB Images**: [read pdf](https://arxiv.org/pdf/1705.01389v3.pdf)  
  Christian Zimmermann, Thomas Brox

- **Survey on depth and RGB image-based 3D hand shape and pose estimation**: [read pdf](https://www.sciencedirect.com/science/article/pii/S2096579621000280)  
  Lin HUANG , Boshen ZHANG, Zhilin GUO , Yang XIAO , Zhiguo CAO , Junsong YUAN

- **3D Hand Shape and Pose Estimation from a Single RGB Image**: [read pdf](https://arxiv.org/pdf/1903.00812.pdf)  
  Liuhao Ge , Zhou Ren , Yuncheng Li , Zehao Xue, Yingying Wang , Jianfei Cai , Junsong Yuan

- **Pushing the Envelope for RGB-based Dense 3D Hand Pose Estimation via Neural Rendering**: [read pdf](https://arxiv.org/pdf/1904.04196.pdf)  
  Seungryul Baek, Kwang In Kim, Tae-Kyun Kim
    
- **Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild**: [read pdf](https://arxiv.org/pdf/2004.01946.pdf)  
  Dominik Kulon, Riza Alp Guler,  Iasonas Kokkinos, Michael Bronstein, Stefanos Zafeiriou

- **End-to-end Hand Mesh Recovery from a Monocular RGB Image**: [read pdf](https://arxiv.org/pdf/1902.09305.pdf)  
  Xiong Zhang , Qiang Li , Hong Mo , Wenbo Zhang , Wen Zheng

- **3D Hand Shape and Pose Estimation based on 2D Hand Keypoints**: [read pdf](https://dl.acm.org/doi/pdf/10.1145/3594806.3594838)  
  Drosakis Drosakis, Antonis Argyros

- **MobileHand: Real-time 3D Hand Shape and Pose Estimation from Color Image**: [read pdf](https://www.researchgate.net/publication/347025951_MobileHand_Real-Time_3D_Hand_Shape_and_Pose_Estimation_from_Color_Image)  
  Guan Ming Lim, Prayook Jatesiktat, Wei Tech Ang

## Datasets

| **Dataset**             | **Year** | **Type**    | **Synthetic/ Real** | **Object interaction** | **Number of Joints** | **View** | **Number of Subjects** | **Number of Frames**      | **Paper**        | **License**                                      |
|-------------------------|----------|-------------|--------------------|------------------------|-------------|----------|---------------|------------------|------------------|---------------------------------------------------|
| [**HIU-DMTL-Data**](https://github.com/MandyMo/HIU-DMTL/)      | 2021     | RGB         | Real               | no                     | 21          | 3rd/ego  | 200           | 40               | ICCV 2021 [\[PDF\]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Hand_Image_Understanding_via_Deep_Multi-Task_Learning_ICCV_2021_paper.pdf)                 | [MIT](https://github.com/MandyMo/HIU-DMTL/blob/main/LICENSE)                                              |
| [**InterHand2.6M**](https://mks0601.github.io/InterHand2.6M/)       | 2020     | RGB         | Real               | no                     | 21          | 3rd      | 27            | 2.6M             | ECCV 2020 [\[PDF\]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650545.pdf)                                                                                 | [CC-BY-NC 4.0](https://github.com/facebookresearch/InterHand2.6M#license)                                     |
| [**YouTube 3D Hands**](https://github.com/arielai/youtube_3d_hands)    | 2020     | RGB         | Real               | yes                    | -           | 3rd      | -| 47,125/ 1525/1525 | CVPR 2020 [\[PDF\]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kulon_Weakly-Supervised_Mesh-Convolutional_Hand_Reconstruction_in_the_Wild_CVPR_2020_paper.pdf) | [Non-Commercial](https://github.com/snap-research/arielai_youtube_3d_hands/blob/master/LICENSE)      |
| [**OneHand10K**](https://yangangwang.com/papers/WANG-MCC-2018-10.html)         | 2019     | RGB         | Real               | no                     | 21          | 3rd      | 1             | 10k/1.3k         | TCSVT 2019 [\[PDF\]](https://yangangwang.com/papers/WANG-MCC-2018-10.pdf) | [Non-Commercial](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)                                     |
| [**FreiHAND**](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)            | 2019     | RGB         | Real               | yes                    | 21          | 3rd      | -             | 130k/3960        | ICCV 2019 [\[PDF\]](https://arxiv.org/pdf/1909.04349.pdf)                                                                                                                  | [Research Only](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)                                     |
| [**GANerated Hands**](https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/GANeratedDataset.htm)     | 2018     | RGB         | Synthetic          | both                   | 21          | ego      | -             | 330k             | CVPR 2018 [\[PDF\]](https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/content/GANeratedHands_CVPR2018.pdf)                                                        | [Scientific / Non-commercial](https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/GANeratedDataset.htm)                       |
| [**CMU Panoptic HandDB**](http://domedb.perception.cs.cmu.edu/handdb.html) | 2017     | RGB         | Real + Synthetic   | no                     | 21          | 3rd      | -             | 14,817           | CVPR 2017 [\[PDF\]](https://arxiv.org/pdf/1704.07809)                                                                                                                      |      |
| [**MHP**](http://www.rovit.ua.es/dataset/mhpdataset/)                 | 2017     | RGB         | Real               | no                     | 21          | 3rd      | 9             | 80k              | IVC 2017 [\[PDF\]](https://arxiv.org/pdf/1707.03742.pdf)                                                                                                                   | [BSD](http://www.rovit.ua.es/dataset/mhpdataset/license.txt)                                                             |
| [**MVHand**](https://github.com/ShichengChen/multiviewDataset)              | 2021     | RGB + Depth | Real               | no                     | 21          | 3rd      | 4             | 83k              | BMVC 2021 [\[PDF\]](https://arxiv.org/pdf/2112.06389.pdf)                                                                          | [MIT](https://github.com/ShichengChen/multiviewDataset/blob/main/LICENSE)|
| [**ContactPose**](https://contactpose.cc.gatech.edu/)         | 2020     | RGB + Depth | Real               | yes                    | 21          | 3rd      | 50            | 2.9M             | ECCV 2020 [\[PDF\]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580358.pdf)                                                                            | [MIT](https://github.com/facebookresearch/ContactPose/blob/main/LICENSE.txt)                                               |
| [**Ego3DHands**](https://github.com/AlextheEngineer/Ego3DHands)          | 2020     | RGB + Depth | Synthetic          | no                     | 21          | ego      | 1             | 50k/5k           | arXiv 2020 [\[PDF\]](https://arxiv.org/pdf/2006.01320.pdf)                                                                                                            | [Non-Commercial / Scientific only](https://github.com/AlextheEngineer/Ego3DHands#license)                 |
| [**ObMan**](https://www.di.ens.fr/willow/research/obman/data/)               | 2019     | RGB + Depth | Synthetic          | yes                    | -           | -        | -             | 150k            | CVPR 2019 [\[PDF\]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hasson_Learning_Joint_Reconstruction_of_Hands_and_Manipulated_Objects_CVPR_2019_paper.pdf) | [Non-Commercial / Scientific](https://www.di.ens.fr/willow/research/obman/data/requestaccess.php)                      |
| [**EgoDexter**](http://handtracker.mpi-inf.mpg.de/projects/OccludedHands/EgoDexter.htm)           | 2017     | RGB + Depth | Real               | yes                    | 5           | ego      | 4             | 1485             | ICCV 2017 [\[PDF\]](http://handtracker.mpi-inf.mpg.de/projects/OccludedHands/content/OccludedHands_ICCV2017.pdf)                                                      | [Non-Commerical / Scientific](https://handtracker.mpi-inf.mpg.de/projects/OccludedHands/EgoDexter.htm)                     |
| [**SynthHands**](http://handtracker.mpi-inf.mpg.de/projects/OccludedHands/SynthHands.htm)          | 2017     | RGB + Depth | Synthetic          | both                   | 21          | ego      | 2             | 63,53            | ICCV 2017 [\[PDF\]](http://handtracker.mpi-inf.mpg.de/projects/OccludedHands/content/OccludedHands_ICCV2017.pdf)                                                      | [Non-Commercial / Scientific](https://handtracker.mpi-inf.mpg.de/projects/OccludedHands/SynthHands.htm)                       |
| [**RHD**](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)                 | 2017     | RGB + Depth | Synthetic          | no                     | 21          | 3rd      | 20            | 41k/2.7k         | ICCV 2017 [\[PDF\]](https://arxiv.org/pdf/1705.01389.pdf)                                                                                                             | [Research Only](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)                                     |
| [**STB**](https://sites.google.com/site/zhjw1988/)                 | 2017     | RGB + Depth | Real               | no                     | 21          | 3rd      | 1             | 18k              | ICIP 2017 [\[PDF\]](http://www.cs.cityu.edu.hk/~jianbjiao2/pdfs/icip.pdf)                                                                                             | [MIT](https://github.com/zhjwustc/icip17_stereo_hand_pose_dataset/blob/master/LICENSE)                                               |
| [**Dexter+Object**](http://handtracker.mpi-inf.mpg.de/projects/RealtimeHO/dexter+object.htm)       | 2016     | RGB + Depth | Real               | yes                    | 5           | 3rd      | 2             | 3,014           | ECCV 2016 [\[PDF\]](http://handtracker.mpi-inf.mpg.de/projects/RealtimeHO/content/RealtimeHO_ECCV2016.pdf)                                                            | No restriction mentioned. check manually.[link](https://handtracker.mpi-inf.mpg.de/projects/RealtimeHO/dexter+object.htm) |
| [**UCI-EGO**](http://pascal.inrialpes.fr/data2/grogez/UCI-EGO/UCI-EGO.tar.gz)             | 2014     | RGB + Depth | Real               | no                     | 26          | ego      | 2             | 400              | ECCVW 2014 [\[PDF\]](https://www.cs.cmu.edu/~deva/papers/egocentric_depth_workshop.pdf)                                                                               | No restriction mentioned in the file downloaded.  |
| [**Dexter1**](http://handtracker.mpi-inf.mpg.de/projects/handtracker_iccv2013/dexter1.htm)             | 2013     | RGB + Depth | Real               | no                     | 6           | 3rd      | 1             | 2,137            | ICCV 2013 [\[PDF\]](http://handtracker.mpi-inf.mpg.de/projects/handtracker_iccv2013/content/handtracker_iccv2013.pdf)                                                 | No restriction Mentioned.[link](https://handtracker.mpi-inf.mpg.de/projects/handtracker_iccv2013/dexter1.htm)                     |

**PREDOMINANT DATASETS**  
Some of them may appear at the above table under a different name
1. Rendered Hand Dataset (RHD)
2. Stereo Hand Pose Tracking Benchmark (STB)
3. Dexter (Dexter+Object EgoDexter)
4. MPII+NZSL
5. FreiHAND
6. Panoptic (PAN)

Comment: Last update was at January 2024. You may also check [Awesome Hand Pose Estimation](https://github.com/xinghaochen/awesome-hand-pose-estimation) for more updated information on hand pose estimation papers and datasets
## Hand parametric models

**Mano**  
- [Project page](https://mano.is.tue.mpg.de/)
- [MANO paper](https://ps.is.mpg.de/uploads_file/attachment/attachment/392/Embodied_Hands_SiggraphAsia2017.pdf)

<!-- [BlendPose: A New Linear Parametric Hand Model for High-quality 3D Hand Reconstruction](https://dl.acm.org/doi/abs/10.1145/3579109.3579126)  -->
**Other hand parametric models based on MANO:**  
- **HTML**
  A Parametric Hand Texture Model for 3D Hand Reconstruction and Personalization
  - [Project page](https://handtracker.mpi-inf.mpg.de/projects/HandTextureModel/)
  - [HTML paper](https://handtracker.mpi-inf.mpg.de/projects/HandTextureModel/content/HandTextureModel_ECCV2020.pdf)
- **PIANO**
  A Parametric Hand Bone Model from Magnetic Resonance Imaging
  - [Project page](https://liyuwei.cc/proj/piano)
  - [PIANO paper](https://www.ijcai.org/proceedings/2021/0113.pdf)

## Real-time 2D/3D applications

- [MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)  
  The MediaPipe Hand Landmarker task lets you detect the landmarks of the hands in an image.
- [MMPose](https://mmpose.readthedocs.io/en/latest/demos.html#hand-keypoint-estimation)  
  MMPose is a Pytorch-based pose estimation open-source toolkit, a member of the OpenMMLab Project
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)  
  OpenPose has represented the first real-time multi-person system to jointly detect human body, hand, facial, and foot keypoints (in total 135 keypoints) on single images.  
- [Hand Gesture Recognition with YOLOv8 on OAK-D in Near Real-Time](https://pyimagesearch.com/2023/05/15/hand-gesture-recognition-with-yolov8-on-oak-d-in-near-real-time/)  
  Tutorial to perform hand gesture recognition using YOLOv8 on the OAK-D platform  
- [Real-Time 2D and 3D Hand Pose Estimation from RGB Image](https://github.com/enghock1/Real-Time-2D-and-3D-Hand-Pose-Estimation/)  
  Project improving the CVPR 2019 paper ["3D Hand Shape and Pose Estimation from a Single RGB Image"](https://github.com/3d-hand-shape/hand-graph-cnn)  
- [Real-time Hand-Detection using Neural Networks (SSD) on Tensorflow.](https://github.com/victordibia/handtracking)  
  Documentation of steps and scripts used to train a hand detector using Tensorflow (Object Detection API).
- [MobileHand: Real-time 3D Hand Shape and Pose Estimation from Color Image](https://gmntu.github.io/mobilehand/)  
  An approach for real-time estimation of 3D hand shape and pose from a single RGB image.
