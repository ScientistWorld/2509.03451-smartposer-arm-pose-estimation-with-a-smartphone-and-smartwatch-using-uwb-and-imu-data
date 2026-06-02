# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

SmartPoser estimates arm pose using only an off-the-shelf smartphone and smartwatch by combining inertial measurements with phone-watch UWB distance. The central claim is that absolute UWB range stabilizes and improves arm tracking compared with IMU-only sensing.

## Approach

The system has two learned stages trained end to end. First, a 2-layer bidirectional LSTM with 8 hidden units per direction corrects noisy raw UWB range using 125-frame windows of distance, phone/watch orientation matrices, and accelerations. Second, a 2-layer bidirectional LSTM with 32 hidden units per direction consumes corrected range plus the same IMU features and predicts a 125-frame sequence of shoulder, elbow, and wrist positions normalized by arm span. Training uses Adam with learning rate 3e-4 and a summed loss: UWB distance MSE plus mean per-joint position error. At inference, the 120th frame of each 125-frame output window is used for a 200 ms latency/accuracy tradeoff.

## Main Claims

Primary: phone/watch UWB plus IMU data can track wrist and elbow pose around 11 cm median error without training on the target user. Secondary: the UWB correction network reduces structured range error, and the complete UWB+IMU system outperforms an IMU-only ablation.
