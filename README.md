# ARoFace: Alignment Robustness to Improve Low-quality Face Recognition
European Conference on Computer Vision (ECCV 2024).

**[Saeed Ebrahimi](https://msed-ebrahimi.github.io/)★, [Sahar Rahimi](https://scholar.google.com/citations?user=EIwfFsQAAAAJ&hl=en&oi=ao)★, [Ali Dabouei](https://alldbi.github.io/), [Nasser Nasrabadi](https://scholar.google.com/citations?user=PNhUilMAAAAJ&hl=en)**

★ Equal contribution

Aiming to enhance Face Recognition (FR) on Low-Quality (LQ) inputs, recent studies suggest incorporating synthetic LQ samples into training. Although promising, the quality factors that are considered in these works are general rather than FR-specific, e.g., atmospheric turbulence, resolution, etc.
  Motivated by the observation of the vulnerability of current FR models to even small Face Alignment Errors (FAE) in LQ images, we present a simple yet effective method that considers FAE as another quality factor that is tailored to FR. We seek to improve LQ FR by enhancing FR models' robustness to FAE. To this aim, we formalize the problem as a combination of differentiable spatial transformations and adversarial data augmentation in FR. We perturb the alignment of the training samples using a controllable spatial transformation and enrich the training with samples expressing FAE.
  We demonstrate the benefits of the proposed method by conducting evaluations on IJB-B, IJB-C, IJB-S (+4.3% Rank1), and TinyFace (+2.63%)

* We introduce Face Alignment Error (FAE) as an image degradation factor tailored for FR which has previously been ignored in LQ FR studies.
* We propose an optimization method that is specifically tailored to increase the FR model robustness against FAE.
* We show that the proposed optimization can greatly increase the FR performance in real-world LQ evaluations such as IJB-S and TinyFace. Moreover, our framework achieves these improvements without sacrificing the performance on datasets with both HQ and LQ samples such as IJB-B and IJB-C.
* We empirically show that the proposed method is a plug-and-play module, providing an orthogonal improvement to SOTA FR methods.

![Demo](assets/fig1.jpg)
Visual comparison of aligned (a) and alignment-perturbed (b) samples from the IJB-B dataset. (c, d, e) 
The performance difference between aligned inputs and those with slight FAE.
Models exhibit robustness to FAE in HQ samples but suffer significant performance drops in LQ faces, with over 50% reduction in TAR@FAR=1e-5. Results from two distinct ResNet-100 trained on MS1MV3 using ArcFace/AdaFace objective.

## TinyFace Evaluations

<table>
  <tr>
    <th rowspan="1">Method</th>
    <th colspan="1">Training Set</th>
    <th colspan="1">Rank1</th>
    <th colspan="1">Rank5</th>
  </tr>
  <tr style="background-color: #f2f2f2;">
    <td>URL</td>
    <td>MS1MV2</td>
    <td>63.89</td>
    <td>68.67</td>
  </tr>
  <tr style="background-color: #f2f2f2;">
    <td>CurricularFace</td>
    <td>MS1MV2</td>
    <td>63.68</td>
    <td>67.65</td>
  </tr>
  <tr style="background-color: #f2f2f2;">
    <td>ArcFace+CFSM</td>
    <td>MS1MV2</td>
    <td>64.69</td>
    <td>68.80</td>
  </tr>
  <tr style="background-color: #f2f2f2;">
    <td><b>ArcFace+ARoFace</b></td>
    <td>MS1MV2</td>
    <td><b>67.32</b></td>
    <td><b>72.45</b></td>
  </tr>
  <tr style="background-color: #e6e6e6;">
    <td>ArcFace</td>
    <td>MS1MV3</td>
    <td>63.81</td>
    <td>68.80</td>
  </tr>
  <tr style="background-color: #e6e6e6;">
    <td><b>ArcFace+ARoFace</b></td>
    <td>MS1MV3</td>
    <td><b>67.54</b></td>
    <td><b>71.05</b></td>
  </tr>
  <tr style="background-color: #f2f2f2;">
    <td>AdaFace</td>
    <td>WebFace4M</td>
    <td>72.02</td>
    <td>74.52</td>
  </tr>
  <tr style="background-color: #f2f2f2;">
    <td><b>AdaFace+ARoFace</b></td>
    <td>WebFace4M</td>
    <td><b>73.98</b></td>
    <td><b>76.47</b></td>
  </tr>
  <tr style="background-color: #e6e6e6;">
    <td>AdaFace</td>
    <td>WebFace12M</td>
    <td>72.29</td>
    <td>74.97</td>
  </tr>
  <tr style="background-color: #e6e6e6;">
    <td><b>AdaFace+ARoFace</b></td>
    <td>WebFace4M</td>
    <td><b>74.00</b></td>
    <td><b>76.87</b></td>
  </tr>
</table>

## IJB-S Evaluations

<table>
  <tr>
    <th rowspan="1">Method</th>
    <th colspan="1">Venue</th>
    <th colspan="1">Dataset</th>
    <th colspan="1">Surveillance-to-Single Rank1</th>
    <th colspan="1">Surveillance-to-Single Rank5</th>
    <th colspan="1">Surveillance-to-Single 1</th>
    <th colspan="1">Surveillance-to-Booking Rank1</th>
    <th colspan="1">Surveillance-to-Booking Rank5</th>
    <th colspan="1">Surveillance-to-Booking 1</th>
    <th colspan="1">Surveillance-to-Surveillance Rank1</th>
    <th colspan="1">Surveillance-to-Surveillance Rank5</th>
    <th colspan="1">Surveillance-to-Surveillance 1</th>
  </tr>
  
  <tr style="background-color: #f2f2f2;">
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
  <tr style="background-color: #f2f2f2;">
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
  <tr style="background-color: #f2f2f2;">
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
  <tr style="background-color: #f2f2f2;">
    <td><b>ArcFace+ARoFace</b></td>
    <td></td>
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
  
  <tr style="background-color: #e6e6e6;">
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
  <tr style="background-color: #e6e6e6;">
    <td><b>ArcFace+ARoFace</b></td>
    <td></td>
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
  
  <tr style="background-color: #f2f2f2;">
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
  <tr style="background-color: #f2f2f2;">
    <td><b>AdaFace+ARoFace</b></td>
    <td></td>
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



## Table of Contents
- [Usage](#usage)
- [Balanced Classification Results](#balanced-classification-results)
- [Long-tailed Classification Results](#long-tailed-classification-results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)



## Usage
### Prototype Estimation
One can generate equidistributed prototypes with desired dimension:
```
python Prototype_Estimation.py --seed 100 --num_centroids 100 --batch_size 100 --space_dim 50 --num_epoch 1000
```
Also, you can find the estimated prototype in [link](Estimated_prototypes/)
### Training classifier
The configs can be found in ./config/Blanced or LongTail/FILENAME.yaml.
```
python train.py --cfg {path to config}
```

## Balanced Classification Results

<table>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="4">CIFAR-10</th>
    <th colspan="4">ImageNet-200</th>
  </tr>
  <tr>
    <td>d=10</td>
    <td>d=25</td>
    <td>d=50</td>
    <td>d=100</td>
    <td>d=25</td>
    <td>d=50</td>
    <td>d=100</td>
    <td>d=200</td>
  </tr>
  <tr>
    <td>PSC</td>
    <td>25.67</td>
    <td>60.0</td>
    <td>60.6</td>
    <td>62.1</td>
    <td>60.0</td>
    <td>60.6</td>
    <td>62.1</td>
    <td>33.1</td>
  </tr>
  <tr>
    <td>Word2Vec</td>
    <td>29.0</td>
    <td>44.5</td>
    <td>54.3</td>
    <td>57.6</td>
    <td>44.5</td>
    <td>54.3</td>
    <td>57.6</td>
    <td>30.0</td>
  </tr>
  <tr>
    <td>HPN</td>
    <td>51.1</td>
    <td>63.0</td>
    <td>64.7</td>
    <td><b>65.0</b></td>
    <td>63.0</td>
    <td>64.7</td>
    <td><b>65.0</b></td>
    <td><b>44.7</b></td>
  </tr>
  <tr>
    <td><b>Ours</b></td>
    <td><b>57.21</b></td>
    <td><b>64.63</b></td>
    <td><b>66.22</b></td>
    <td>62.85</td>
    <td><b>64.63</b></td>
    <td><b>66.22</b></td>
    <td>62.85</td>
    <td>37.28</td>
  </tr>
</table>


#### ImageNet-1K Classification Accuracy (%) when $d=512$:

| Method               | Venue        | Backbone   | Optimizer | Accuracy (%) |
|----------------------|--------------|------------|-----------|--------------|
| PSC                  | CVPR 2016    | ResNet-50  | SGD       | 76.51        |
| DNC                  | ICLR 2022    | ResNet-50  | SGD       | 76.49        |
| Goto et al.          | WACV 2024    | ResNet-50  | SGD       | 77.19        |
| Kasarla et al.       | NeurIPS 2022 | ResNet-50  | SGD       | 74.80        |
| **Ours**             | CVPR 2024    | ResNet-50  | SGD       | **77.47**    |
| DNC                  | ICLR 2022    | ResNet-101 | SGD       | 77.80        |
| Goto et al.          | WACV 2024    | ResNet-101 | SGD       | 78.27        |
| Kasarla et al.       | NeurIPS 2022 | ResNet-152 | SGD       | 78.50       |
| **Ours**             | CVPR 2024            | ResNet-101 | SGD       | **79.63**    |
| PSC                  | CVPR 2016    | Swin-T     | AdamW     | 76.91        |
| **Ours**             | CVPR 2024          | Swin-T     | AdamW     | **77.26**    |

## Long-tailed Classification Results

<table>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="3">CIFAR-10 LT (d=64)</th>
    <th colspan="3">SVHN LT (d=64)</th>
    <th colspan="3">STL-10 LT (d=64)</th>
  </tr>
  <tr>
    <td>0.005</td>
    <td>0.01</td>
    <td>0.02</td>
    <td>0.005</td>
    <td>0.01</td>
    <td>0.02</td>
    <td>0.005</td>
    <td>0.01</td>
    <td>0.02</td>
  </tr>
  <tr>
    <td>PSC</td>
    <td>67.3</td>
    <td>72.8</td>
    <td>78.6</td>
    <td>40.5</td>
    <td>40.9</td>
    <td>49.3</td>
    <td>33.1</td>
    <td><b>37.9</b></td>
    <td><b>38.8</b></td>
  </tr>
  <tr>
    <td>ETF</td>
    <td><b>71.9</b></td>
    <td>76.5</td>
    <td>81.0</td>
    <td><b>42.8</b></td>
    <td>45.7</td>
    <td><b>49.8</b></td>
    <td>33.5</td>
    <td>37.2</td>
    <td>37.9</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>71.5</td>
    <td><b>76.9</b></td>
    <td><b>81.4</b></td>
    <td>40.9</td>
    <td><b>47.0</b></td>
    <td>49.7</td>
    <td><b>35.7</b></td>
    <td>35.6</td>
    <td>38.0</td>
  </tr>
</table>

#### CIFAR-100 LT Classification Accuracy (%):

| Method |  d  | 0.005  | 0.01   | 0.02   |
|--------|:---:|:------:|:------:|:------:|
| PSC    | 128 |  38.7  |  43.0  |  48.1  |
| ETF    | 128 | *40.9* | **45.3** |  50.4  |
| **Ours** | 128 | **41.3** |  44.9  | *50.7* |



### Citation
```
```

## Acknowledgments

Here are some great resources we benefit from:

* [MUNIT](https://github.com/NeuralCollapseApplications/ImbalancedLearning) for the Long-Tail classification.
* [HPN](https://github.com/psmmettes/hpn) and [EBV](https://github.com/aassxun/Equiangular-Basis-Vectors) for the Balanced classification. 
* [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://github.com/SsnL/align_uniform) for Prototype Estimation.

## Contact
If there is a question regarding any part of the code, or it needs further clarification, please create an issue or send me an email: me00018@mix.wvu.edu.
