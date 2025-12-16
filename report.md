\documentclass[11pt]{article}
% ============================
% Page layout (2 cm margins)
% ============================
\usepackage[margin=2cm]{geometry}

% ============================
% Encoding and fonts
% ============================
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}

% ============================
% Useful packages
% ============================
\usepackage{graphicx}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{url}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}

% ============================
% Space optimization
% ============================
\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\abovecaptionskip}{4pt}
\setlength{\belowcaptionskip}{2pt}
\raggedbottom

% ============================
% Document begins
% ============================
\begin{document}

\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}

% ============================
% ITU Front Page (EXCLUDED from 10 pages)
% ============================
\begin{titlepage}
\centering

\vspace*{1cm}
\includegraphics[width=0.6\textwidth]{ITU_logo_DK jpg.jpg}\\[2cm]

{\LARGE\textbf{Machine Learning}}\\[0.3cm]
{\large\textbf{BSMALEA1KU}}\\[2cm]

\textbf{Matteo Cannata}\\
matca@itu.dk\\[2cm]

\vfill
IT University of Copenhagen\\[0.2cm]
\today
\end{titlepage}

% ============================
% From here on: 10-page limit
% ============================

\section{Introduction}

This project explores the problem of predicting claims risk in automobile insurance. The goal is to estimate the expected claim frequency for each policy based on vehicle and driver (ITU, 2025).

I implemented two methods from scratch (a decision tree regressor and a feed-forward neural network) and used scikit-learn (Scikit-learn, 2025) for a third method (random forest). All models were evaluated using 5-fold cross-validation during hyperparameter tuning, with the held-out test set reserved exclusively for final evaluation. The report describes the data preparation process, the implementation details of each method, and a comparative analysis of their performance.

\section{Target Variable \& Evaluation Metrics}

\subsection{Target Variable}

The goal of this project is to model the \textbf{claims risk} associated with each insurance policy (ITU, 2025). Since the dataset includes both the number of claims (ClaimNb) and the amount of time the policy was active (Exposure), I defined the target variable as \textbf{ClaimFrequency} $= \frac{\text{ClaimNb}}{\text{Exposure}}$.

This choice reflects the idea that a fair measure of risk should account not only for how many claims occurred, but also for how long the driver was insured. A policy active for only a few months cannot be compared directly to a policy active for a full year. By normalising the number of claims by exposure time, the model learns a quantity that is comparable across all policies.

I also considered alternative definitions. Using raw \texttt{ClaimNb} would ignore exposure entirely, penalising short policies unfairly. Using a binary indicator (claim vs.\ no claim) would discard information about multiple claims. The frequency-based target keeps all available information while producing a quantity that is directly interpretable as expected claims per year.

\subsection{Evaluation Metrics}

Since \textbf{ClaimFrequency} is a continuous target, I use four regression metrics. \textbf{Mean Absolute Error (MAE)} $= \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$ measures the average absolute deviation and is robust to outliers, making it the primary metric for hyperparameter selection given the extreme values in the target. \textbf{Root Mean Squared Error (RMSE)} $= \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$ penalises large errors more heavily and is expressed in the same units as the target, aiding interpretability. \textbf{Mean Squared Error (MSE)} is simply RMSE$^2$ and serves as the loss function during training. Finally, \textbf{R\textsuperscript{2}} $= 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ indicates the proportion of variance explained by the model (James et al., 2023).


\section{Data Cleaning and Exploratory Analysis}

The dataset was already in good shape: all features were fully populated, no duplicates were present, and each policy identifier (\texttt{IDpol}) was unique. The only variable requiring correction was \texttt{Exposure}, where a small number of values exceeded the theoretical upper bound of one year. These rows were removed to keep the data consistent with the definition of time at risk. The exploratory analysis highlighted modelling challenges from the start: heavy zero-inflation of the target, rare but valid extreme frequencies, and weak linear relationships between features and claims.


\subsection*{Exposure Values}

To verify that all variables were consistent with their intended meaning, I looked closely at the distribution of \texttt{Exposure}, which represents the fraction of the year during which a policy was active. Before any cleaning, the exposure values ranged from 0.0027 to 2.0100, and the histogram clearly showed that almost all policies fell within the expected interval $[0, 1]$, with a noticeable spike at exactly~1 year. In total, 134{,}483 policies (24.79\%) had full-year exposure.

A small number of observations, 994 rows, corresponding to about 0.18\% of the dataset, had \texttt{Exposure > 1}. Because exposure cannot logically exceed one year in this setting, these values fell outside the valid range (ITU, 2025). I therefore removed all rows with \texttt{Exposure > 1}. This was the only feature that required an explicit cleaning step.

After removing these entries, the training set contained 541{,}416 rows, and the exposure range became $[0.0027, 1.0000]$. This correction also ensured that the target variable, defined as \texttt{ClaimFrequency = ClaimNb / Exposure}, would not be distorted by invalid exposure values. The distribution of exposure after cleaning matches the expected interpretation of time at risk and provides a reliable basis for subsequent modelling.

\begin{figure}[!htb]
\centering
\includegraphics[width=0.65\textwidth]{exposure.png}
\vspace{-0.5em}
\caption{Distribution of \textbf{Exposure} values in the dataset.}
\label{fig:exposure_distribution}
\end{figure}


\subsection*{ClaimNb and the Distribution of Claims}

The variable \texttt{ClaimNb} represents the number of claims recorded for each policy during the observation year. Its distribution is extremely imbalanced. Out of all 541{,}416 policies in the cleaned training set, 514{,}246 (94.98\%) reported no claims at all, while 27{,}170 policies (5.02\%) had at least one claim. The maximum observed value is 11 claims, although such high counts are very rare. The vast majority of non-zero values are equal to~1, with frequencies dropping sharply for larger counts.

All values fall within a plausible range for annual insurance claims, so no cleaning was required for this feature. Instead, its highly skewed structure simply reflects the challenge of modelling rare events.


\subsection*{Behaviour of the Target Variable}

After cleaning exposure, I constructed the target variable \texttt{ClaimFrequency = ClaimNb / Exposure}. The distribution of this target highlights two distinct characteristics. First, there is a very large proportion of zeros, which reflects the fact that the majority of drivers do not file any claims within the year. Second, among the positive values, the distribution has a long right tail. Extremely high frequencies occur when a claim happens over a very short exposure period. These values are not errors; they naturally arise from the definition of the target and represent genuinely high-risk observations that the model should be able to learn from.

\begin{figure}[!htb]
    \centering
\includegraphics[width=0.65\textwidth]{claimFreq.png}
\vspace{-0.5em}
    \caption{Distribution of \texttt{ClaimFrequency} on a logarithmic scale.}
\end{figure}


\subsection*{Numerical Feature Distributions}

For the remaining numerical variables (\texttt{VehPower}, \texttt{VehAge}, \texttt{DrivAge}, \texttt{Density}, and \texttt{BonusMalus}), I inspected summary statistics, histograms, and outlier counts. All features fall within plausible ranges. A few extreme values were visible in \texttt{VehAge} and \texttt{DrivAge}, for example, occasional entries equal to 100, but these cases are extremely rare. Because they do not violate any domain rules, I kept these observations and documented them as natural extremes rather than errors.

\subsection*{Categorical Feature Structure}

The dataset includes four categorical variables: \texttt{Area}, \texttt{VehBrand}, \texttt{VehGas}, and \texttt{Region}. These features differ in both the type of information they encode and the structure of their categories. The variable \texttt{Area} is \textbf{ordinal}, since its six levels (A--F) represent increasing levels of urbanisation from rural to urban. In contrast, \texttt{VehBrand}, \texttt{VehGas}, and \texttt{Region} are \textbf{nominal} variables whose categories have no inherent ordering. \texttt{VehBrand} distinguishes between eleven vehicle brands, \texttt{VehGas} indicates the fuel type, and \texttt{Region} identifies one of twenty-two geographical regions.

All categorical variables are complete, with no missing values or impossible category labels. The distribution of categories is generally well balanced, and even the smaller categories appear plausible within the context of insurance data. Based on this inspection, no categorical values required cleaning or correction.



\subsection*{Relationships and Correlations}

Figure~\ref{fig:correlations} shows the correlation matrix for the numerical features together with their
individual correlations with the target variable \texttt{ClaimFrequency}. All numerical variables exhibit
extremely weak linear relationships with the target. The strongest positive correlation is observed for
\texttt{BonusMalus} ($r = 0.0139$), while the strongest negative correlation is found for \texttt{VehAge}
($r = -0.0240$). No feature pair shows substantial collinearity, and no correlations exceed $|r| > 0.5$.

\begin{figure}[!htb]
    \centering
\includegraphics[width=0.65\textwidth]{correlations.png}
\vspace{-0.5em}
    \caption{Correlation matrix of numerical features}
    \label{fig:correlations}
\end{figure}

To assess the effect of categorical variables on claim frequency, one-way ANOVA tests were conducted.
All variables show statistically significant differences between categories:

\texttt{Area} ($p = 7.5 \times 10^{-6}$), \texttt{VehBrand} ($p = 1.4 \times 10^{-87}$), \texttt{VehGas} ($p = 3.7 \times 10^{-16}$), and \texttt{Region} ($p = 2.9 \times 10^{-15}$).

Although statistically significant, these features each explain less than $0.1\%$ of the overall variance
in \texttt{ClaimFrequency}, reflecting the heavy zero-inflation and noise in the target variable.
Nevertheless, relative differences exist: for example, Area~F shows higher claim frequency than Area~A
(risk ratio $\approx 1.78\times$), and Region~R21 exhibits higher risk compared to Region~R83
(risk ratio $\approx 3.44\times$). These effects, while modest in absolute terms, highlight differences across categories.

\section{Feature Engineering \& Preprocessing}

The transformations applied in this project were intentionally kept simple. Since the dataset was already well structured and required minimal cleaning, the main goal was to prepare the inputs in a way that respected their meaning and made them suitable for the modelling steps that followed.

The only genuine feature engineering step was the construction of the target variable \texttt{ClaimFrequency = ClaimNb / Exposure}, defined after removing the few invalid exposure values. No further adjustments were made to this quantity, because even its extreme values arise naturally when a claim occurs over a very short exposure period and therefore carry meaningful information.

All remaining transformations fall under preprocessing. For the input features, I followed the natural distinction between numerical, ordinal, and nominal variables. The feature \texttt{Area} is ordinal, with levels A--F representing increasing levels of urbanisation, so it was encoded as integers from 0 to~5. The nominal categorical variables \texttt{VehBrand}, \texttt{VehGas}, and \texttt{Region} do not have an inherent order, so they were one-hot encoded whenever a numerical representation was required (for example in PCA, K-Means, the neural network, the random forest, and the reference scikit-learn decision tree (Scikit-learn, 2025)). The five numerical features (\texttt{VehPower}, \texttt{VehAge}, \texttt{DrivAge}, \texttt{Density}, and \texttt{BonusMalus}) were kept in their original form, and scaling was applied only when needed by methods that are sensitive to feature magnitudes, such as PCA, K-Means, and the neural network.

No additional feature engineering techniques were introduced. The intention was to preserve the original structure of the dataset while ensuring that the learning algorithms received inputs in an appropriate and consistent numerical format.

To implement the preprocessing steps, I used tools from scikit-learn (Scikit-learn, 2025). Numerical scaling was performed with \texttt{StandardScaler}, which standardises each numerical feature to zero mean and unit variance. The ordinal feature \texttt{Area} was encoded with \texttt{OrdinalEncoder}, using the explicit category order \texttt{['A','B','C','D','E','F']}. The nominal variables were transformed with \texttt{OneHotEncoder}, using \texttt{drop='first'} to avoid redundant columns and \texttt{sparse\_output=false} to return a dense array.
 
\section{PCA \& Clustering}

\subsection{Numerical Features Only}

Before introducing the categorical variables, I first focused on understanding the structure of the numerical features alone. This provides a cleaner view of how the continuous variables relate to each other and whether they contain any natural lower-dimensional structure (James et al., 2023). It also helps reveal whether claim outcomes show any separation when projected onto the main directions of variation in the data.

To do this, I applied PCA to the five scaled numerical features (\texttt{VehPower}, \texttt{VehAge}, \texttt{DrivAge}, \texttt{Density}, and \texttt{BonusMalus}). The first two principal components captured just over half of the total variance (approximately 52\%), while including the third raised this to about 72\%. This indicates that the numerical features do not collapse cleanly into a single dominant direction, but instead distribute their variability across several dimensions.

Because the full dataset contains more than 540{,}000 observations, plotting every point would make the figures visually dense and difficult to interpret. For this reason, I sampled 10{,}000 observations uniformly at random for all PCA and clustering visualisations. This is large enough to capture the overall shape of the numerical feature space while keeping the plots readable.

Figure~\ref{fig:pca_numerical_claims} shows the PCA projection coloured by claim status. The points form one large cloud without any obvious separation between policies with and without claims. The red points (non-zero claims) are scattered throughout the entire space, suggesting that the numerical variables on their own do not create a visibly separable structure that distinguishes risky from non-risky policies.

\begin{figure}[!htb]
    \centering
\includegraphics[width=0.65\textwidth]{PCA numerical.png}
\vspace{-0.5em}
    \caption{PCA of numerical features coloured by claim status (10{,}000-point subsample).}
    \label{fig:pca_numerical_claims}
\end{figure}

I then applied K-Means clustering directly in the PCA space, using the elbow curve and silhouette scores to guide the choice of $K$. Both diagnostics pointed towards $K=3$ as a reasonable balance between compactness and separation. As with PCA, the clustering visualisation uses a 10{,}000-point subsample purely for clarity, while the actual clustering was performed on all 541{,}416 observations.

The resulting clusters are shown in Figure~\ref{fig:kmeans_numerical}. Each cluster is continuous and overlaps with the others, but they do correspond to different average claim frequencies when mapped back to the full dataset. Cluster~0 is the largest group and shows the lowest mean claim frequency (0.24). Cluster~1 has a slightly higher risk level (0.32), and Cluster~2, although small, contains the highest average risk (0.40). All medians remain zero, reflecting the strong zero-inflation of the target variable.

\begin{figure}[!htb]
    \centering
\includegraphics[width=0.65\textwidth]{clustering.png}
\vspace{-0.5em}
    \caption{K-Means clustering ($K=3$) in PCA space based on numerical features (10{,}000-point subsample).}
    \label{fig:kmeans_numerical}
\end{figure}

\subsection{PCA and Clustering on All Features}

After analysing the numerical features on their own, I extended the PCA and clustering to include the categorical variables as well. Since PCA requires numerical inputs, this first meant encoding all categorical features. The ordinal variable \texttt{Area} was encoded according to its natural order from A to~F, while the nominal variables \texttt{VehBrand}, \texttt{VehGas}, and \texttt{Region} were one-hot encoded. This resulted in a complete feature set of 38 numerical inputs.

With all features represented numerically, I applied PCA to explore whether combining numerical and categorical information would reveal clearer structure in the data. As in the numerical-only case, the variance was spread across many components, and no low-dimensional projection captured a dominant share. The scatterplots of the first two principal components again formed a dense cloud of overlapping points. Policies with and without claims appeared intermingled across the space, and no obvious separation emerged between them, even when all available features were included.

I then applied K-Means clustering to the complete feature set. The elbow curve showed steady but modest reductions in inertia as $K$ increased (for example, inertia was $3{,}785{,}919$ at $K=2$, $3{,}369{,}930$ at $K=3$, and $3{,}022{,}633$ at $K=4$). The silhouette scores were consistently low (0.1720 at $K=2$, 0.1722 at $K=3$, and 0.1681 at $K=4$), indicating that the clusters were not strongly separated. For consistency with the numerical-only analysis, I selected $K=3$ here as well. The resulting clusters overlapped and differed only modestly in their average claim frequencies (0.22, 0.27, and 0.35 for the three clusters).

Compared to the numerical-only clustering, the silhouette scores for the full feature set were notably lower (around 0.17 instead of roughly 0.29). This suggests that adding many one-hot encoded dimensions made it harder for K-Means to form compact, well-separated groups, this is because the additional binary features introduced noise rather than clear structure. Overall, extending PCA and clustering to all features did not reveal distinct high-risk or low-risk clusters; instead, the data remained diffuse, with differences in claim behavior spread gradually across many dimensions.

\section{Methods}

\subsection{Decision Tree Regressor (M1)}

For the first method, I implemented a decision tree regressor from scratch using only NumPy. The implementation supports both numerical and categorical features natively, meaning the tree can split on categorical variables directly without requiring one-hot encoding. This is useful because it preserves the structure of features like \texttt{VehBrand} and \texttt{Region}, where one-hot encoding would inflate the feature space from 9 to 38 dimensions.

The tree-building algorithm follows the standard recursive approach: at each node, it searches for the split that minimises the mean squared error. For numerical features, it considers all possible thresholds using sorted unique values. For categorical features, it uses a mean-based ordering strategy that groups categories by their average target value, then finds the best binary split along this ordering. This approach is optimal for the MSE criterion and avoids testing all possible category subsets.

Regularisation is controlled by three hyperparameters: \texttt{max\_depth}, \texttt{min\_samples\_split}, and \texttt{min\_samples\_leaf}. Tuning was performed using 5-fold cross-validation on the training set, with MAE as the selection criterion. First, I tested four depth values (3, 5, 7, and 10) while keeping \texttt{min\_samples\_split=100} and \texttt{min\_samples\_leaf=50}. The results showed that deeper trees achieved lower MAE: depth=10 gave MAE $= 0.4786 \pm 0.0094$, compared to $0.4835 \pm 0.0077$ for depth=3. I then fixed depth=10 and tested different combinations of \texttt{min\_samples\_split} and \texttt{min\_samples\_leaf}: (100, 50), (1000, 500), (5000, 2500), and (10000, 5000). The first combination performed best, so the final configuration was \texttt{max\_depth=10}, \texttt{min\_samples\_split=100}, \texttt{min\_samples\_leaf=50}.

To verify correctness, I ran a set of dedicated tests. These included checking that the tree correctly splits on the most predictive feature, that it handles categorical features properly, and that the number of unique predictions matches the number of leaves. I also compared the from-scratch implementation against scikit-learn's \texttt{DecisionTreeRegressor} (Scikit-learn, 2025) using the same hyperparameters. Both implementations produced nearly identical MAE values (difference of 0.00021), confirming that the from-scratch version behaves as expected.

\subsection{Feed-Forward Neural Network (M2)}

For the second method, I implemented a feed-forward neural network from scratch using only NumPy. The implementation supports arbitrary layer architectures, ReLU activations in hidden layers, and a linear output layer for regression. Training uses mini-batch stochastic gradient descent with backpropagation. Weights are initialised using He initialisation, and biases are set to zero.

Because neural networks require numerical inputs and are sensitive to feature scales, I applied different preprocessing than for the decision tree. Numerical features were standardised using \texttt{StandardScaler}, ordinal encoding was applied to \texttt{Area}, and one-hot encoding to the remaining categorical variables. This produced 38 input features.

Hyperparameter tuning followed a sequential search strategy using 5-fold cross-validation with MAE as the selection metric. In the first stage, I compared three architectures while keeping learning rate fixed at 0.001 and epochs at 100: [38,64,32,1], [38,32,32,16,1], and [38,128,64,1]. The wider architecture [38,64,32,1] achieved the lowest MAE ($0.4672 \pm 0.0358$). In the second stage, I fixed this architecture and varied the learning rate (0.01 and 0.0001). The original learning rate of 0.001 remained best. In the third stage, I tested 200 epochs, but this did not improve over 100. The final configuration was architecture [38,64,32,1], learning rate 0.001, and 100 epochs.

To verify correctness, I ran several tests: checking that the output shape is correct, that the loss decreases during training, that the network can learn both linear and nonlinear functions ($R^2 > 0.98$ on synthetic data), and that the MSE loss and its derivative are computed correctly. I also compared against scikit-learn's \texttt{MLPRegressor} (Scikit-learn, 2025) with equivalent settings. On this dataset, the from-scratch implementation actually achieved better MAE (0.4808 vs.\ 0.5437 for scikit-learn), likely due to differences in optimisation and early stopping behaviour.

\subsection{Random Forest (M3)}

For the third method, I chose Random Forest, an ensemble of decision trees that combines bagging with feature randomisation. This method was implemented using scikit-learn's \texttt{RandomForestRegressor} (Scikit-learn, 2025).

Random Forest addresses some limitations of a single decision tree by training many trees on bootstrap samples and averaging their predictions. Each tree also considers only a random subset of features at each split, which decorrelates the trees and reduces variance.

The preprocessing was similar to the neural network but without scaling, since tree-based methods are invariant to feature magnitudes. The input consisted of 38 encoded features.

Hyperparameter tuning tested six configurations that progressively increased regularisation, evaluated using 5-fold cross-validation with MAE as the selection criterion. The configurations were: (1)~baseline with default parameters, (2)~\texttt{min\_samples\_leaf=10}, (3)~adding \texttt{max\_depth=15}, (4)~adding \texttt{min\_samples\_split=50}, (5)~adding \texttt{max\_features=0.5}, and (6)~full regularisation with all constraints plus \texttt{max\_samples=0.6}. The baseline achieved MAE $= 0.5267$, while adding regularisation steadily improved this. The best configuration was \texttt{max\_depth=15}, \texttt{min\_samples\_leaf=10}, \texttt{max\_features=0.5}, achieving MAE $= 0.4767 \pm 0.0090$.

Since this method relies on scikit-learn's well-tested implementation (Scikit-learn, 2025), correctness is assured by the library itself. I verified that the model was used correctly by checking that feature importance values summed to approximately~1 and that the number of trees matched the specified \texttt{n\_estimators}. The cross-validation results were also consistent across folds, confirming stable behaviour.

\section{Results}

After tuning, I trained each model on the full training set and evaluated on the held-out test set. Table~\ref{tab:results} summarises the results.

\begin{table}[H]
\centering
\caption{Test set performance of all three methods.}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{MAE} & \textbf{RMSE} & \textbf{R\textsuperscript{2}} \\
\midrule
Decision Tree     & 0.4922 & 5.2032 & 0.0005 \\
Neural Network    & 0.4806 & 5.1856 & 0.0073 \\
Random Forest     & 0.4886 & 5.1751 & 0.0113 \\
\bottomrule
\end{tabular}
\end{table}


\begin{figure}[!htb]
\centering
\includegraphics[width=0.9\textwidth]{comparisons.png}
\vspace{-0.5em}
\caption{Comparison of model performance across evaluation metrics.}
\label{fig:comparison}
\end{figure}

\section{Discussion}

The results confirm what the exploratory analysis suggested: predicting claim frequency at the individual level is very hard. All three methods achieved R\textsuperscript{2} values close to zero on the test set, explaining almost none of the variance in the target. This reflects the problem itself: the target is heavily zero-inflated (over 94\% have no claims), and positive values span a wide range up to 732. The available features capture risk at a coarse level but lack the behavioural information needed to predict individual claims.

Among the three methods, the Neural Network achieved the best MAE (0.4806), suggesting its flexibility allows it to learn a prediction function that minimises average error more effectively. The Random Forest achieved the highest R\textsuperscript{2} (0.0113), capturing slightly more variance through its ensemble averaging, but also showed the largest overfitting gap (train R\textsuperscript{2} of 0.096 vs.\ test R\textsuperscript{2} of 0.011). This indicates that despite regularisation, the ensemble memorised some training patterns that did not generalise. The Decision Tree showed the smallest gap (0.026), generalising best relative to its training performance, which reflects its simpler hypothesis space.

Why do the models perform so similarly despite their different architectures? The PCA and clustering analysis showed that policies with and without claims are not separable in feature space. The neural network's nonlinear transformations, the random forest's ensemble diversity, and the decision tree's recursive partitioning all face the same fundamental limitation: when the signal is weak, model complexity cannot extract patterns that are not present in the features.

Each method has distinct characteristics worth noting. The from-scratch decision tree handles categorical features natively using mean-based ordering, avoiding the 9-to-38 dimension expansion required by one-hot encoding. The neural network benefits from its ability to learn smooth, continuous functions, which may explain its lower MAE. The random forest's bootstrap aggregation reduces variance but cannot overcome the lack of predictive signal in the features.

Regularisation was essential for all models. Without constraints, they overfit by memorising noise. For decision trees, limiting depth to 10 and requiring at least 50 samples per leaf prevented overly specific splits. For random forest, the baseline MAE of 0.5267 improved to 0.4767 after tuning with depth limits and feature subsampling. Training times ranged from 9 seconds (decision tree) to 45 seconds (neural network), with random forest at 17 seconds, reflecting the computational trade-offs between model complexity and training cost.

\section{Conclusion}

This project explored three machine learning methods for modelling claims risk: a decision tree and neural network implemented from scratch, plus scikit-learn's random forest (Scikit-learn, 2025). All methods achieved similar performance (MAE between 0.48--0.49, R\textsuperscript{2} below 0.012), consistent with the known difficulty of predicting insurance claims. The from-scratch implementations were validated against scikit-learn references, with the decision tree matching within 0.0002 MAE and the neural network outperforming sklearn's MLPRegressor. Hyperparameter tuning via 5-fold cross-validation improved all models, with the random forest showing the largest gain.

The main takeaway is that standard regression struggles with this data. The weak correlations observed in the EDA (all $|r| < 0.03$) and the lack of cluster separation in PCA space foreshadowed the low predictive performance. For future work, specialised methods would be more appropriate: Poisson regression handles the discrete nature of claims, zero-inflated models account for the 94\% of policies with no claims, and two-stage approaches separate the "will a claim occur?" question from "how frequent?". These methods match the data structure better than standard regression, which treats the target as continuous and ignores the excess of zeros (James et al., 2023).

\section*{References}

Anthropic (2025) \textit{Claude 4.0 Opus}. Available at: \url{https://www.anthropic.com/} (Accessed: December 2025).

Bishop, C.M. (2006) \textit{Pattern Recognition and Machine Learning}. New York: Springer.

IT University of Copenhagen (2025) \textit{Exam Assignment: Machine Learning (BSc Data Science), Fall 2025}. Copenhagen: IT University of Copenhagen.

James, G., Witten, D., Hastie, T., Tibshirani, R. and Taylor, J. (2023) \textit{An Introduction to Statistical Learning: With Applications in Python}. New York: Springer.

Scikit-learn (2025) \textit{Scikit-learn: Machine Learning in Python}. Available at: \url{https://scikit-learn.org/stable/} (Accessed: December 2025).

\subsection*{Tools}

Claude 4.0 Opus (Anthropic, 2025) was used for code completion, syntax correction, and optimisation guidance during development.

\end{document}
