# MPDC：Multi-Factor Driven Ideal Location Preferences for Next POI Recommendations

This paper proposes a novel next POI recommendation method (**MPDC**) that leverages multi-perspective modeling of visit factors and diffusion-based contrastive learning. We mitigate non-subjective visit factors by leveraging bidirectional information from user sequences and designing a preference filter. Meanwhile, we integrate spatio-temporal information and build a spatial-aware hyperbolic graph module to consider users' subjective visit factors. Based on the generation capability of diffusion models, we develop a diffusion-based preference contrast module that generates latent ideal location preferences for users under different context conditions guided through a diffusion model. 

# Environment
    pip install python==3.11.4
    pip install torch-geometric == 2.5.3
    pip install numpy==1.24.0
    pip install pytorch ==2.0.1
    pip install scipy==1.11.2

Please refer to the **requirements.txt** for specific environmental requirements.

# Data


| Dataset | User | POI | Check-ins| Avg.Visit| Density|
| --------| --------| ------ |--------|----|------|
| **LA**  | 965     | 2,541  |37,181  | 39 | 1.5% |
| **NYC** | 1,083   | 9,989  |179,468 | 166| 1.7% |
| **SP**  | 3,722   | 12,829 |287,642 | 77 | 0.6% |
| **JK**  | 5,358   | 10,706 |300,324 | 56 | 0.5% |
| **IST** | 9,208   | 11,871 |529,067 | 57 | 0.5% |


* We have updated processed data in the './data/processed'. 
* For the raw data processing, we follow the Diff-POI data processing method.
* Due to the large file limit of GitHub, we cannot directly upload processed data of other dataset. We will update all of them in other link for more convenience after acceptance.

# Running

You can run this code on the uploaded LA dataset:

    python main.py --dataset la --batch 1024 --patience 10 --dropout --gpu 0 --keepprob 0.7
