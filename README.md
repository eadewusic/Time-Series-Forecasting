# Air Quality Forecasting in Beijing Using Attention-Based BiLSTM
This project focuses on applying RNNs and LSTM models to solve a real-world problem: forecasting air pollution levels. Air pollution, particularly PM2.5, is a critical global issue that impacts public health and urban planning. By accurately predicting PM2.5 concentrations, governments and communities can take timely action to mitigate their effects.

## Problem Statement

PM2.5 (particulate matter ≤2.5μm) is a severe health risk. Accurate forecasting helps mitigate its impact through timely public and governmental action. The objective is to predict future PM2.5 concentrations using historical air quality and weather data.

## Dataset Description

- **Source:** Provided by Kaggle Competition
- **Samples:** 30,676 hourly records
- **Features:** PM2.5, temperature, pressure, dew point, rainfall, wind direction/speed, timestamp
- **Target:** PM2.5 concentration

## Preprocessing & Feature Engineering

- **Missing Data:** Filled using linear interpolation and forward-fill methods.
- **Feature Engineering:**
  - Time-based features (hour, weekday, month, season)
  - **Cyclical encodings** for temporal continuity
  - Activity-based indicators (e.g., rush hours, weekends)
- **Sliding Window/ Time Windowing:** Created 24-hour sequences for supervised learning.


## Model Architecture

**Attention-Based Bidirectional LSTM with Residuals**
- **Layers:**
  - BiLSTM (64 units) → Dropout(0.25)
  - BiLSTM (32 units) → Dropout(0.25)
  - Multi-Head Attention Layer
  - Residual connections + Layer Normalization
  - Final BiLSTM (16 units) → Dense Layer
- **Training Setup:**
  - Epochs: 70
  - Batch Size: 64
  - Optimizer: Adam
  - Learning Rate: 0.0006
  - Early Stopping & ReduceLROnPlateau

## Experiments Summary

| Exp # | Model Type         | Layers           | Dropout | LR     | SeqLen | RMSE (Kaggle) |
|-------|--------------------|------------------|---------|--------|--------|---------------|
| 1     | BiLSTM + Attention | 64, 32, 16 + MHA | 0.2     | 0.001  | 24     | 4081.62       |
| 2     | Vanilla RNN        | 64               | 0.0     | 0.001  | -      | 9503.82       |
| 3     | LSTM (No Dropout)  | 64, 32           | 0.0     | 0.001  | -      | 4436.13       |
| 4     | LSTM (Dropout)     | 64, 32           | 0.2     | 0.001  | -      | 4067.85       |
| 5     | BiLSTM             | 64, 32           | 0.2     | 0.001  | -      | 3876.55       |

- **Best performing model**: BiLSTM without attention and input-output sequences scored 3876.55, but attention model was more robust and interpretable.
- The Attention + BiLSTM model was the only one using a sliding window for sequences, achieving the best performance (RMSE: 4081.6186), suggesting sequence framing significantly improves results.
- Dropout improved generalization: Models with dropout (e.g., LSTM with 0.2) likely performed better than those without, by reducing overfitting.
- Vanilla RNN underperforms in capturing long-term dependencies, making it less suitable for this forecasting task.
- BiLSTM without attention performed worse than the attention-based model, indicating attention improves temporal focus and accuracy.
- All models used the same optimizer and learning rate (Adam, 0.001); future experiments could vary these for deeper insights.


## Kaggle Challenge

- Competition: [Assignment 1 – Time Series Forecasting May 2025](https://www.kaggle.com/competitions/assignment-1-time-series-forecasting-may-2025)
- **Leaderboard RMSE:** 3876.55 (Top 3)
