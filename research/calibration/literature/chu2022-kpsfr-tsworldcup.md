# Sports Field Registration via Keypoints-aware Label Condition (KpSFR)
- Authors: Yen-Jui Chu, Jheng-Wei Su, Kai-Wen Hsiao, Chi-Yu Lien, Shu-Ho Fan, Min-Chun Hu, Ruen-Rone Lee, Chih-Yuan Yao, Hung-Kuo Chu
- Venue: CVPR Workshops (CVsports) 2022
- URL: https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.html · code/data https://ericsujw.github.io/KpSFR/

## Key findings
- 91 uniformly spaced field keypoints, detected as instance segmentation with dynamic filters (DoDNet head on a ResNet-34 U-Net); homography by DLT + RANSAC (threshold 10).
- Introduces **TS-WorldCup**: 3,812 time-sequence images from 43 videos (WC 2014/2018), 2,925 train / 887 test, with GT homographies.
- Metrics: IoU_part, IoU_entire, projection error (metres, 2,500 sampled visible-field pixels projected with predicted vs GT homography), re-projection error.
- TS-WorldCup test: Nie et al. 0.57 / 0.51 m (mean/median), KpSFR 0.45 / 0.41 m; fine-tuned 0.36 / 0.33 m. WC14 test: KpSFR 0.81 / 0.63 m.

## Relevance
Gives us a temporal benchmark with homography ground truth and a metric in metres, the unit our speed and position outputs use. Sets the bar: well under 1 m mean error is achievable.
