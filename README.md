# 506-midterm

Our project aimed to build an efficient and accurate model for predicting review scores based on text data and associated helpfulness metrics. Due to computational constraints, we limited our training to a 20% sample of the data, assuming this subset would be representative enough to capture key patterns. The final model leverages a stacking ensemble approach, combining feature engineering and two base models—RandomForest and Logistic Regression—with an XGBoost meta-model to improve classification accuracy. This approach provided the flexibility to capture diverse relationships in the data without overfitting.

In the preprocessing phase, we focused on extracting meaningful features from the text and numeric data provided in the dataset. To enhance the model’s ability to capture relevant patterns, we engineered several custom features, including a Helpfulness ratio (derived from HelpfulnessNumerator and HelpfulnessDenominator), word count, and average word length. Additionally, we applied CountVectorizer to identify the top 10 keywords from reviews with high helpfulness scores and supplemented them with manually selected keywords such as "love," "great," and "recommend." These keywords, encoded as binary features, provided insights into sentiment and common patterns among different review scores.

For model training, we adopted a two-level stacking ensemble. At the base level, we trained a RandomForestClassifier with 75 trees and a LogisticRegression model, each trained on cross-validation splits to capture variability. Their probability outputs were then used as inputs for the meta-model. The final meta-model, an XGBoostClassifier, was tuned using grid search to find optimal parameters, including n_estimators, learning_rate, max_depth, and reg_lambda. By calculating the mean and maximum probabilities from the base models as features, the meta-model was able to capture underlying trends from both linear and non-linear relationships, enhancing accuracy without excessive complexity.

In conclusion, this layered approach allowed us to effectively utilize a reduced dataset while maximizing predictive performance. Keyword-based features, combined with word count and helpfulness ratios, proved invaluable in distinguishing review scores. Additionally, tuning the stacking meta-model enabled it to generalize well on the unseen test data, yielding a competitive accuracy. This ensemble model, with its focus on meaningful feature engineering and balanced model complexity, provided an effective solution for the problem within the computational limits imposed.
