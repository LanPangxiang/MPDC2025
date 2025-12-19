# MPDC：Multi-Factor Driven Expected Location Preferences for Next POI Recommendations

This paper proposes a novel next POI recommendation method (\textbf{MPDC}) that leverages \textbf{M}ulti-\textbf{P}erspective modeling and \textbf{D}iffusion-based \textbf{C}ontrastive learning. From the behavior-aware perspective, a behavior-aware graph prompt module mitigates external contextual influences via bidirectional sequence modeling and a preference filter, while capturing visit preferences. From the spatial-aware perspective, a spatial-aware hyperbolic graph module models higher-order, nonlinear geographic patterns to better reflect visiting preferences. Additionally, a diffusion-based preference contrast module models users' expected check-in intentions under different context conditions. Leveraging diffusion models' generative capacity, it simulates latent expected location preferences aligned with user intent rather than just observed destinations.

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
* All datasets are sourced from https://sites.google.com/site/yangdingqi/home/foursquare-dataset
* For the raw data processing, we follow the Diff-POI data processing method.
* Due to the large file limit of GitHub, we cannot directly upload processed data of other dataset. We will update all of them in other link for more convenience after acceptance.

# Running

You can run this code on the uploaded LA dataset:

    python main.py --dataset la --batch 1024 --patience 10 --dropout --gpu 0 --keepprob 0.7
