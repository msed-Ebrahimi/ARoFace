# ARoFace: Alignment Robustness to Improve Low-quality Face Recognition
European Conference on Computer Vision (ECCV 2024).

**[Saeed Ebrahimi](https://msed-ebrahimi.github.io/)★, [Sahar Rahimi](https://saharr1372.github.io/)★, [Ali Dabouei](https://alldbi.github.io/), [Nasser Nasrabadi](https://scholar.google.com/citations?user=PNhUilMAAAAJ&hl=en)**

★ Equal contribution

> Aiming to enhance Face Recognition (FR) on Low-Quality (LQ) inputs, recent studies suggest incorporating synthetic LQ samples into training. Although promising, the quality factors that are considered in these works are general rather than FR-specific, e.g., atmospheric turbulence, resolution, etc.
  Motivated by the observation of the vulnerability of current FR models to even small Face Alignment Errors (FAE) in LQ images, we present a simple yet effective method that considers FAE as another quality factor that is tailored to FR. We seek to improve LQ FR by enhancing FR models' robustness to FAE. To this aim, we formalize the problem as a combination of differentiable spatial transformations and adversarial data augmentation in FR. We perturb the alignment of the training samples using a controllable spatial transformation and enrich the training with samples expressing FAE.
  We demonstrate the benefits of the proposed method by conducting evaluations on IJB-B, IJB-C, IJB-S (+4.3% Rank1), and TinyFace (+2.63%)

**[Arxive](https://arxiv.org/abs/2407.14972), [HuggingFace](https://huggingface.co/papers/2407.14972)

![Demo](assets/fig1.jpg)
Visual comparison of aligned (a) and alignment-perturbed (b) samples from the IJB-B dataset. (c, d, e) 
The performance difference between aligned inputs and those with slight FAE.
Models exhibit robustness to FAE in HQ samples but suffer significant performance drops in LQ faces, with over 50% reduction in TAR@FAR=1e-5. Results from two distinct ResNet-100 trained on MS1MV3 using ArcFace/AdaFace objective.


* We introduce Face Alignment Error (FAE) as an image degradation factor tailored for FR which has previously been ignored in LQ FR studies.
* We propose an optimization method that is specifically tailored to increase the FR model robustness against FAE.
* We show that the proposed optimization can greatly increase the FR performance in real-world LQ evaluations such as IJB-S and TinyFace. Moreover, our framework achieves these improvements without sacrificing the performance on datasets with both HQ and LQ samples such as IJB-B and IJB-C.
* We empirically show that the proposed method is a plug-and-play module, providing an orthogonal improvement to SOTA FR methods.


## TinyFace Evaluations

<table>
  <tr style="background-color: #dee2e6;">
    <th rowspan="1">Method</th>
    <th colspan="1">Training Set</th>
    <th colspan="1">Rank1</th>
    <th colspan="1">Rank5</th>
  </tr>
  <tr style="background-color: #ffffff;">
    <td>URL</td>
    <td>MS1MV2</td>
    <td>63.89</td>
    <td>68.67</td>
  </tr>
  <tr style="background-color: #ffffff;">
    <td>CurricularFace</td>
    <td>MS1MV2</td>
    <td>63.68</td>
    <td>67.65</td>
  </tr>
  <tr style="background-color: #ffffff;">
    <td>ArcFace+CFSM★</td>
    <td>MS1MV2</td>
    <td>64.69</td>
    <td>68.80</td>
  </tr>
  <tr style="background-color: #ffffff;">
    <td><b>ArcFace+ARoFace</b></td>
    <td>MS1MV2</td>
    <td><b>67.32</b></td>
    <td><b>72.45</b></td>
  </tr>
  <tr style="background-color: #dee2e6;">
    <td>ArcFace</td>
    <td>MS1MV3</td>
    <td>63.81</td>
    <td>68.80</td>
  </tr>
  <tr style="background-color: #dee2e6;">
    <td><b>ArcFace+ARoFace</b></td>
    <td>MS1MV3</td>
    <td><b>67.54</b></td>
    <td><b>71.05</b></td>
  </tr>
  <tr style="background-color: #ffffff;">
    <td>AdaFace★</td>
    <td>WebFace4M</td>
    <td>72.02</td>
    <td>74.52</td>
  </tr>
  <tr style="background-color: #ffffff;">
    <td><b>AdaFace+ARoFace</b></td>
    <td>WebFace4M</td>
    <td><b>73.98</b></td>
    <td><b>76.47</b></td>
  </tr>
  <tr style="background-color: #dee2e6;">
    <td>AdaFace</td>
    <td>WebFace12M</td>
    <td>72.29</td>
    <td>74.97</td>
  </tr>
  <tr style="background-color: #dee2e6;">
    <td><b>AdaFace+ARoFace</b></td>
    <td>WebFace4M</td>
    <td><b>74.00</b></td>
    <td><b>76.87</b></td>
  </tr>
</table>

★ Re-runs with official code due to missing trained checkpoints on the specified dataset in the official repository

## IJB-S Evaluations

<table>
  <tr style="background-color: #dee2e6;">
    <th rowspan="2">Method</th>
    <th rowspan="2">Venue</th>
    <th rowspan="2">Dataset</th>
    <th colspan="3">Surveillance-to-Single</th>
    <th colspan="3">Surveillance-to-Booking</th>
    <th colspan="3">Surveillance-to-Surveillance</th>
  </tr>
<tr style="background-color: #dee2e6;">
    <td>Rank1</td>
    <td>Rank5</td>
    <td>1</td>
    <td>Rank1</td>
    <td>Rank5</td>
    <td>1</td>
    <td>Rank1</td>
    <td>Rank5</td>
    <td>1</td>
  </tr>
  
  <tr style="background-color: #ffffff;">
    <td>ArcFace</td>
    <td>CVPR2019</td>
    <td>MS1MV2</td>
    <td>57.35</td>
    <td>64.42</td>
    <td>41.85</td>
    <td>57.36</td>
    <td>64.95</td>
    <td>41.23</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="background-color: #ffffff;">
    <td>PFE</td>
    <td>ICCV2019</td>
    <td>MS1MV2</td>
    <td>50.16</td>
    <td>58.33</td>
    <td>31.88</td>
    <td>53.60</td>
    <td>61.75</td>
    <td>35.99</td>
    <td>9.20</td>
    <td>20.82</td>
    <td>0.84</td>
  </tr>
  <tr style="background-color: #ffffff;">
    <td>URL</td>
    <td>ICCV2020</td>
    <td>MS1MV2</td>
    <td>59.79</td>
    <td>65.78</td>
    <td>41.06</td>
    <td>61.98</td>
    <td>67.12</td>
    <td>42.73</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr style="background-color: #ffffff;">
    <td><b>ArcFace+ARoFace</b></td>
    <td>ECCV2024</td>
    <td>MS1MV2</td>
    <td><b>61.65</b></td>
    <td><b>67.6</b></td>
    <td><b>47.87</b></td>
    <td><b>60.66</b></td>
    <td><b>67.33</b></td>
    <td><b>46.34</b></td>
    <td><b>18.31</b></td>
    <td><b>32.07</b></td>
    <td><b>2.23</b></td>
  </tr>
  
  <tr style="background-color: #dee2e6;">
    <td>ArcFace</td>
    <td>CVPR2019</td>
    <td>WebFace4M</td>
    <td>69.26</td>
    <td>74.31</td>
    <td>57.06</td>
    <td>70.31</td>
    <td>75.15</td>
    <td>56.89</td>
    <td>32.13</td>
    <td>46.67</td>
    <td>5.32</td>
  </tr>
  <tr style="background-color: #dee2e6;">
    <td><b>ArcFace+ARoFace</b></td>
    <td>ECCV2024</td>
    <td>WebFace4M</td>
    <td><b>70.96</b></td>
    <td><b>75.54</b></td>
    <td><b>58.67</b></td>
    <td><b>71.70</b></td>
    <td><b>75.24</b></td>
    <td><b>58.06</b></td>
    <td><b>32.95</b></td>
    <td><b>50.30</b></td>
    <td><b>6.81</b></td>
  </tr>
  
  <tr style="background-color: #ffffff;">
    <td>AdaFace</td>
    <td>CVPR2022</td>
    <td>WebFace12M</td>
    <td>71.35</td>
    <td>76.24</td>
    <td>59.40</td>
    <td>71.93</td>
    <td>76.56</td>
    <td>59.37</td>
    <td>36.71</td>
    <td>50.03</td>
    <td>4.62</td>
  </tr>
  <tr style="background-color: #ffffff;">
    <td><b>AdaFace+ARoFace</b></td>
    <td>ECCV2024</td>
    <td>WebFace12M</td>
    <td><b>72.28</b></td>
    <td><b>77.93</b></td>
    <td><b>61.43</b></td>
    <td><b>73.01</b></td>
    <td><b>79.11</b></td>
    <td><b>60.02</b></td>
    <td><b>40.51</b></td>
    <td><b>50.90</b></td>
    <td><b>6.37</b></td>
  </tr>
</table>

## Usage
### Training sets
Download and prepare datasets from [InsightFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) repository
### Training
The total batch size we used for training was 2048 on four Nvidia RTX 6000 ADA.
To have stable training, choose the learning rate based on the total batch size on your machine:
```
config.lr = (0.1*config.batch_size*config.ngpus)/(1024)
```
Please modify the 
```
config.ngpus = 4
```
according to your resources in [configs](configs).

Then, for training on one machine using four GPUs:
```
torchrun --nproc_per_node=4 train_v2.py configs/ms1mv2_r100
```
# Pretrained Models

| Method          | Arch | Dataset   | Link                                                                                          |
|-----------------|------|-----------|-----------------------------------------------------------------------------------------------|
| ArcFace+ARoFace | R100 | MS1MV2    | [link](https://drive.google.com/file/d/1dB407DQXYBN16pySRA0q012b2ww2CnHz/view?usp=drive_link) |
| ArcFace+ARoFace | R100  | MS1MV3    | [link](https://drive.google.com/file/d/1z_me8OshifKuLv1znk9-peTQr3pwTMUQ/view?usp=drive_link) |
| ArcFace+ARoFace | R100 | WebFace4M | [link](https://drive.google.com/file/d/1ro-x-pLGpiiQjW0jIdZAIilGQeqxRW8j/view?usp=drive_link) |
| AdaFace+ARoFace | R100 | WebFace4M | [link](https://drive.google.com/file/d/1I9dtPc_753wSMVHtkwu57RttP-Obh3Ce/view?usp=drive_link) |
| AdaFace+ARoFace | R100 | WebFace12M | [link](https://drive.google.com/file/d/1tHqcQBY5s10uxNdfGtIYmYDEKypNtOuQ/view?usp=drive_link) |

### Citation
```
@misc{saadabadi2024arofacealignmentrobustnessimprove,
      title={ARoFace: Alignment Robustness to Improve Low-Quality Face Recognition}, 
      author={Mohammad Saeed Ebrahimi Saadabadi and Sahar Rahimi Malakshan and Ali Dabouei and Nasser M. Nasrabadi},
      year={2024},
      eprint={2407.14972},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.14972}, 
}
```

## Acknowledgments

Here are some great resources we benefit from:

* [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) and [AdaFace](https://github.com/mk-minchul/AdaFace) for the face recognition module. 
* [advertorch](https://github.com/BorealisAI/advertorch), [RobustAdversarialNetwork](https://github.com/DengpanFu/RobustAdversarialNetwork), and [CFSM](https://github.com/liufeng2915/CFSM/tree/main) for the adversarial regularization.
## Contact
If there is a question regarding any part of the code, or it needs further clarification, please create an issue or send me an email: me00018@mix.wvu.edu.
