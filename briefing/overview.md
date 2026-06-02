# Overview

- **Paper ID:** 2509.03451
- **Title:** SmartPoser: Arm Pose Estimation with a Smartphone and Smartwatch Using UWB and IMU Data
- **Domain:** Mobile sensing / wearable pose estimation
- **TL;DR:** SmartPoser fuses phone-watch UWB range with IMU streams to estimate 3-D arm pose using commodity devices.

## Short Summary

The paper targets mobile arm pose estimation without cameras, external beacons, or extra worn sensors. It proposes a two-stage recurrent model: one network corrects noisy UWB range and another predicts shoulder, elbow, and wrist positions from corrected range plus IMU features. The original study reports centimeter-scale tracking on held-out users and shows that adding UWB improves over IMU-only sensing.

## Key Results

- Median elbow+wrist error: 11.0 cm in the combined study data.
- UWB correction reduces mean range error from 21.0 cm to 8.6 cm.
- IMU-only ablation worsens median elbow+wrist error from 11.0 cm to 14.2 cm.
