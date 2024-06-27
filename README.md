# Social Popularity Prediction

This document presents a study on predicting social media popularity, covering preprocessing techniques, model selection, results, and evaluation.

## Quick Links

- [Detailed Analysis Paper](https://github.com/your-repo/Social-Popularity-Prediction/blob/main/Data_Mining_Homework_2.pdf): Detailed analysis and insights are shown in the pdf.

## Environment Setup
- To run the provided code, use Kaggle Notebook or Colab (with necessary file path modifications) to utilize CUDA (GPU) for improved efficiency.
- Ensure you have the necessary libraries installed, including `scikit-learn`, `pandas`, `numpy`, `Pytorch`, and `Catboost`.

## Preprocessing
### Feature Engineering
- **Text Features**: Utilized pre-trained embeddings (`paraphrase-MiniLM-L6-v2`), concept vectors via `word2vec`, and manual features like `tags_count` and `title_len`.
- **Time Features**: Included one-hot encoded cyclical features (`weekday`, `month`, `hour`) and duration features based on `PostDate`.
- **Categorical Features**: Processed features like `Category`, `Subcategory`, `Concept`, `Uid`, and frequency of user posts with PCA reduction where necessary.
- **Normalization**: Applied z-score normalization to numeric features, ensuring no data leakage by calculating mean and standard deviation only on training data. Images were resized to (3, 128, 128).

### Outlier Removal
- This was not explicitly mentioned in the LaTeX code, but standard practice would involve identifying and handling outliers during preprocessing.

## Models
### Model Construction
Three models were discussed and evaluated:
1. **Multimodal with Linear Fusion Layer**: Combined features from `MLP_branch` and `Image_branch` using a linear fusion layer.
2. **Multimodal with LSTM Fusion Layer**: Added a time sequence aspect with LSTM layers for fusion.
3. **Catboost with Pre-trained Resnet50**: Used pre-trained `Resnet50` for image feature extraction, reduced dimensions via SVD, and combined with numerical features for input into Catboost.

### Hyperparameters Tuning
- Applied Bayesian Optimization for Random Forest, XGBoost, and CatBoost.
- Used Random search for Multi-Layer Perceptron (MLP) due to its complex architecture and hyperparameters.

## Results Interpretation
### Model Performance
- Performance evaluated using cross-validation (cv = 3).
- Reported metrics include negative mean absolute error for the folds.

### Empirical Results
- **Feature Importance**: The study discussed the impact of various features on the prediction models.
- **Model Selection**: Random Forest generally outperformed XGBoost and CatBoost. MLP performance was suboptimal.

### Conclusion
- Despite time-consuming methods, results were not ideal. The study noted potential issues with dataset differences and computational constraints.
- Future research directions include dynamic adjustment of computational resources and further exploration of feature engineering techniques.

## References
- Abousaleh, F., Cheng, W., Yu, N., Tsao, Y., Senior Member, IEEE. (2021). "Multimodal Deep Learning Framework for Image Popularity Prediction on Social Media."
- Lv, J., Liu, W., Zhang, M., Gong, H., Wu, B., Ma, H. (2017). "Multi-feature Fusion for Predicting Social Media Popularity."