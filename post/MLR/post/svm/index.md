# SVM

中文：[zh](./index_zh.html)

[TOC]

## Classification SVM Algorithm Theory

### 1. Basic SVM

#### 1.1. Problem Description

The core of classification problems lies in finding a decision boundary that can effectively separate samples of different categories. In a two-dimensional space, this decision boundary can be a line; in three-dimensional space, a plane; and in high-dimensional spaces, a hyperplane. For linearly separable cases, assuming the dataset consists of two classes of labels, the goal is to find an optimal hyperplane that separates the two classes of data and maximizes the distance (i.e., margin) from the samples of both classes to the hyperplane. This not only enhances classification accuracy but also increases the model's robustness to noisy data.

#### 1.2. Working Principle of SVM

The core idea of SVM is to find a hyperplane in the data space that maximizes the margin. This hyperplane is defined by a set of specific sample points (i.e., support vectors), which are the samples closest to the hyperplane. The objective of SVM is to maximize the distance between the hyperplane and the support vectors (i.e., the margin), thereby making the classification model more generalizable.

Assume the dataset contains $N$ samples $(x_i, y_i)$, where $x_i \in \mathbb{R}^d$ represents the feature vector of the $i$-th sample, and $y_i \in \{-1, 1\}$ represents the label of the sample. The goal of SVM is to find a linear decision function:

$$
f(x) = w \cdot x + b
$$

where $w$ is the weight vector and $b$ is the bias term, such that the function $f(x) = 0$ corresponds to the separating hyperplane. For linearly separable data, SVM aims to find the optimal $w$ and $b$ that maximize the margin between the different classes of samples from the separating hyperplane.

#### 1.3. Optimization Objective

To achieve margin maximization, the optimization objective constructed by SVM is as follows:

$$
\text{maximize} \quad M = \frac{2}{\|w\|}
$$

This means maximizing the margin $M$, which is the distance from the support vectors to the hyperplane. Through appropriate transformations, the optimization problem of SVM can be represented as a constrained quadratic optimization problem:

$$
\min \frac{1}{2} \|w\|^2
$$

$$
\text{s.t.} \quad y_i (w \cdot x_i + b) \geq 1, \quad i = 1, 2, \ldots, N
$$

In the above equations, the constraint $y_i (w \cdot x_i + b) \geq 1$ ensures that all sample points are correctly classified under the constraints of the hyperplane, and the distance is not less than 1.

#### 1.4. SVM Solving Methods

The solving method for SVM typically uses the Lagrange multiplier method and KKT (Karush-Kuhn-Tucker) conditions to convert the constrained optimization problem into an unconstrained optimization problem. By introducing Lagrange multipliers $\alpha_i$, the objective function becomes:

$$
L(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^{N} \alpha_i [y_i (w \cdot x_i + b) - 1]
$$

By taking the partial derivatives of $L(w, b, \alpha)$ with respect to $w$ and $b$ and setting them to zero, the dual problem can be obtained. The final dual optimization objective is:

$$
\max \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
$$

$$
\text{s.t.} \quad \sum_{i=1}^{N} \alpha_i y_i = 0, \quad \alpha_i \geq 0, \quad i = 1, 2, \ldots, N
$$

After solving for $\alpha$, the weight vector $w$ and bias $b$ can be calculated as:

$$
w = \sum_{i=1}^{N} \alpha_i y_i x_i
$$

Then, select a support vector $x_k$ to compute the bias:

$$
b = y_k - w \cdot x_k
$$

Once $w$ and $b$ are obtained, we can classify new data points using the decision function $f(x) = \text{sign}(w \cdot x + b)$.

#### 1.5. Summary

The optimization objective of basic SVM is to find a hyperplane that maximizes the margin between classes, thereby enhancing the model's robustness and generalization ability. By solving the dual problem using the Lagrangian dual approach, SVM automatically selects the most influential sample points (support vectors) during training, ultimately obtaining a classification hyperplane.

### 2. Soft Margin SVM

In practical problems, data is often not completely linearly separable and may contain noise points or overlapping regions. To address this issue, SVM introduces the concept of a soft margin. The soft margin SVM allows some misclassified samples around the classification boundary, balancing classification accuracy and the model's generalization ability. By adding a slack variable $\xi$, the objective function of the soft margin SVM can be expressed as:

$$
\min \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i
$$

where $C$ is the penalty parameter that controls the tolerance for misclassification. A larger $C$ value means the model places more emphasis on classification accuracy, striving to minimize misclassifications but may lead to overfitting; a smaller $C$ value tends to enlarge the margin, allowing more misclassifications, thereby enhancing the model's generalization ability.

### 3. Kernel Trick

For non-linearly separable data, SVM uses the kernel trick to map the data into a high-dimensional space to achieve linear separability. In the high-dimensional space, SVM can perform linear separation on complex non-linear data. Common kernel functions include the linear kernel, polynomial kernel, Radial Basis Function (RBF) kernel, and Sigmoid kernel.

The introduction of kernel functions allows SVM to transform non-linear problems into linear ones, greatly expanding the application range of SVM. The common form of kernel functions is:

$$
K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)
$$

where $\phi(x)$ is the mapping function that maps the original feature space to a high-dimensional space. The kernel trick does not require explicit computation of the high-dimensional mapping but directly computes the similarity between features through the kernel function, thus maintaining computational efficiency.

The choice of kernel function and the setting of its parameters directly affect the performance of the SVM classification model and need to be adjusted according to the specific problem's distribution characteristics.

In support vector machines, the main role of the kernel function is to map data from a low-dimensional space to a high-dimensional space, making non-linearly separable data linearly separable in the high-dimensional space. The choice of kernel function has a significant impact on the model's performance; different kernel functions are suitable for different data distributions and features. Below are some common kernel functions and their applicable scenarios:

#### 3.1. Linear Kernel

**Expression**: $ K(x_i, x_j) = x_i \cdot x_j $

**Applicable Scenarios**:  
The linear kernel is the simplest kernel function, suitable for linearly separable datasets. It performs well in low-dimensional spaces or scenarios where the number of features far exceeds the number of samples. For example, high-dimensional sparse feature data such as text classification and image classification typically fit well with the linear kernel. In these applications, the class boundaries are often close to linear distributions, making the linear kernel effective and efficient for classification.

**Advantages and Disadvantages**:  

- **Advantages**: High computational efficiency, especially excellent performance on high-dimensional sparse data.
- **Disadvantages**: Cannot handle non-linear data.

#### 3.2. Polynomial Kernel

**Expression**: $ K(x_i, x_j) = (x_i \cdot x_j + c)^d $

where $c$ is a constant term and $d$ is the degree of the polynomial.

**Applicable Scenarios**:  
The polynomial kernel is suitable for data with complex interaction relationships, but the non-linearity of the data is not obvious. By adjusting the polynomial degree $d$ and the constant term $c$, the polynomial kernel can handle classification problems with certain non-linearity in lower dimensions. It is commonly used in image processing and natural language processing, such as modeling complex relationships between word vectors.

**Advantages and Disadvantages**:  

- **Advantages**: Suitable for moderately non-linear data, flexible in handling different levels of data complexity by adjusting the degree.
- **Disadvantages**: Higher computational cost in high dimensions and large-scale datasets, prone to model overfitting.

#### 3.3. Radial Basis Function (RBF) Kernel or Gaussian Kernel

**Expression**: $ K(x_i, x_j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right) $

where $\sigma$ is a parameter used to adjust the distribution range.

**Applicable Scenarios**:  
The RBF kernel is the most commonly used kernel function, suitable for most non-linear classification problems, especially those with complex feature spaces. It has a localization property, strongly responding to highly similar data points. The RBF kernel is often used in bioinformatics, image recognition, and handwritten digit recognition, among other fields that require capturing complex boundaries.

**Advantages and Disadvantages**:  

- **Advantages**: Capable of flexibly handling highly non-linear classification tasks, strong model generalization ability.
- **Disadvantages**: Sensitive to the parameter $\sigma$; improper parameter settings can easily lead to overfitting or underfitting.

#### 3.4. Sigmoid Kernel

**Expression**: $ K(x_i, x_j) = \tanh(\alpha x_i \cdot x_j + c) $

where $\alpha$ and $c$ are constants, and $\tanh$ is the hyperbolic tangent function.

**Applicable Scenarios**:  
The Sigmoid kernel is somewhat similar to the activation functions in neural networks and is suitable for classification problems with neural network characteristics. It is more commonly used in binary classification tasks, suitable for small-scale, not particularly highly non-linear classification tasks, and with relatively regular data distributions. The Sigmoid kernel can be used for recognizing binary patterns or preliminary experiments on specific classification problems, but its performance is generally not as good as the RBF or polynomial kernels.

**Advantages and Disadvantages**:  

- **Advantages**: Suitable for binary classification tasks, especially in early neural network models.
- **Disadvantages**: Does not necessarily satisfy Mercer's theorem for all kernel functions, so it may not converge in specific scenarios, leading to unstable performance.

#### 3.5. Laplacian Kernel

**Expression**: $ K(x_i, x_j) = \exp\left(-\frac{\|x_i - x_j\|}{\sigma}\right) $

**Applicable Scenarios**:  
The Laplacian kernel is similar to the RBF kernel but uses the L1 distance instead of the L2 distance. It is suitable for scenarios with local similarity and more data noise. It is more common in applications such as signal processing and image segmentation, where sensitivity to local features is required.

**Advantages and Disadvantages**:  

- **Advantages**: More robust to outliers, suitable for noisy data.
- **Disadvantages**: Computational efficiency may be lower, suitable for specific local feature tasks.

The choice of kernel function needs to be adjusted based on the data distribution and the characteristics of the problem. In practical applications, one can start with simple kernel functions (like the linear kernel) and, if the model performance is unsatisfactory, try more complex kernels (like the RBF or polynomial kernels), and optimize the kernel parameters using cross-validation.

## Regression SVM Algorithm Theory

### 1. Basic Regression SVM

#### 1.1. Problem Description

The goal of regression problems is to predict continuous numerical outputs by learning patterns from training data. In Support Vector Regression (SVR), the model aims to find a function such that the prediction error for most data points does not exceed a specified tolerance range $\epsilon$. Unlike classification SVM, SVR no longer focuses on dividing the data into different categories but constructs a "tube" of tolerance error around the regression function, ensuring that most sample points lie within this tube and optimizing to minimize the influence of noise and outliers.

Mathematically, given a dataset $(x_i, y_i)$, SVR attempts to find a linear function:

$$
f(x) = w \cdot x + b
$$

such that for most sample points $(x_i, y_i)$, the difference between the predicted value $f(x_i)$ and the true value $y_i$ does not exceed the tolerance range $\epsilon$. This means the model allows a certain degree of error, but errors beyond this tolerance range are penalized.

#### 1.2. Working Principle of SVR

The core principle of SVR is to construct an $\epsilon$-insensitive zone, which is a margin that allows for some error. This margin is called the "tube" or "regression band." Within this margin, prediction errors are ignored (i.e., no loss is calculated), while errors that exceed this range are penalized.

The optimization problem aims to minimize the model's complexity (by controlling the size of $w$) while ensuring that most sample points are contained within the $\epsilon$-insensitive zone. Specifically, the optimization problem is represented as:

$$
\min \frac{1}{2} \|w\|^2
$$

$$
\text{s.t.} \quad |y_i - (w \cdot x_i + b)| \leq \epsilon
$$

These constraints indicate that all data points should ideally lie within an error range less than $\epsilon$. To further handle samples that exceed the tolerance zone, SVR introduces slack variables $\xi$ and $\xi^*$ to represent deviations in the positive and negative directions:

$$
\text{s.t.} \quad y_i - (w \cdot x_i + b) \leq \epsilon + \xi_i
$$

$$
(w \cdot x_i + b) - y_i \leq \epsilon + \xi_i^*
$$

Finally, the optimization objective of SVR becomes:

$$
\min \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)
$$

where $C$ is the penalty parameter that controls the model's tolerance for errors exceeding $\epsilon$. A larger $C$ value makes the model more focused on reducing errors but may lead to overfitting, while a smaller $C$ value allows more errors, enhancing the model's generalization ability.

#### 1.3. SVR Solving Methods

The core of solving SVR lies in using the Lagrangian dual method to convert the constrained optimization problem into an unconstrained optimization problem. By introducing Lagrange multipliers $\alpha$ and $\alpha^*$, the dual problem can be constructed, simplifying the computation. The optimization process ultimately results in calculating the weight vector $w$ and bias $b$ using support vectors, and the resulting regression model can be used to predict new sample points.

### 2. Application of Kernel Trick in Regression

In practical applications, many datasets are not linearly separable, meaning the relationship between the data and output is not a simple linear relationship. To address this issue, SVR can use kernel functions to map the data into a high-dimensional space, making the regression problem approximately linearly separable in that high-dimensional space. This kernel-based mapping allows for efficient computation without directly calculating high-dimensional coordinates, thereby reducing computational complexity.

Common kernel functions include:

- **Linear Kernel**: Suitable for data with strong linear correlations.
- **Polynomial Kernel**: Suitable for data with complex feature interactions.
- **Radial Basis Function (RBF) Kernel**: Suitable for most non-linear problems, effectively handling local similarities.
- **Sigmoid Kernel**: More commonly used in binary classification tasks with small datasets.

The choice of kernel function directly affects the model's performance and needs to be selected and tuned based on the data characteristics and specific tasks. The Radial Basis Function (RBF) kernel is typically the default choice because its non-linear characteristics are suitable for most real-world applications.

### 3. Parameter Selection and Tuning

The main parameters of SVR include:

- **Penalty Parameter $C$**: Controls the penalty for samples that lie outside the tolerance range. A larger $C$ value makes the model focus more on reducing errors, leading to lower training errors but potentially causing overfitting. A smaller $C$ value allows more errors, thereby enhancing the model's generalization ability.
  
- **Width of the Tolerance Zone $ \epsilon $**: Determines the error range within which the model does not calculate loss. Appropriately increasing $ \epsilon $ can reduce the model's sensitivity to noisy data, thereby improving model stability.
  
- **Kernel Function Parameters (e.g., $ \gamma $ for RBF Kernel)**: Control the feature space mapping of the kernel function, affecting the model's ability to fit non-linear relationships.

In practical use, these parameters typically need to be selected through cross-validation to find the optimal combination for achieving the best regression performance.

## Manual Implementation of SVM Algorithms

In this section, we will manually implement both classification SVM and regression SVM algorithms step by step, writing code tailored to the needs of classification and regression tasks, respectively. The implementation does not use any machine learning libraries but relies solely on basic numerical computation libraries to manually construct the algorithm process, aiding in understanding the core principles and computational processes of SVM algorithms.

### 1. Classification SVM Implementation

The goal of classification SVM is to find an optimal separating hyperplane that maximizes the margin between different classes of samples. Below are the implementation steps and code:

#### 1.1. Initialize Parameters and Kernel Functions

```python
import numpy as np

class SVMClassifier:
    def __init__(self, C=1.0, kernel='linear', gamma=1.0):
        self.C = C  # Penalty coefficient
        self.kernel = kernel  # Type of kernel function
        self.gamma = gamma  # Parameter for RBF kernel

    def linear_kernel(self, X, Y):
        return np.dot(X, Y.T)

    def rbf_kernel(self, X, Y):
        return np.exp(-self.gamma * np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=2) ** 2)

    def kernel_function(self, X, Y):
        if self.kernel == 'linear':
            return self.linear_kernel(X, Y)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X, Y)
```

#### 1.2. Compute Kernel Matrix and Initialize Lagrange Multipliers

```python
def fit(self, X, y):
    n_samples, n_features = X.shape
    self.alpha = np.zeros(n_samples)
    self.b = 0
    self.X_train = X
    self.y_train = y

    # Compute kernel matrix
    K = self.kernel_function(X, X)
```

#### 1.3. Optimize Lagrange Multipliers (Simplified SMO Implementation)

```python
for _ in range(100):  # Set number of iterations
    for i in range(n_samples):
        # Calculate prediction
        prediction = (self.alpha * y) @ K[:, i] + self.b
        # Update alpha_i's value
        error = y[i] * prediction - 1
        if error < 0:
            self.alpha[i] = min(self.C, self.alpha[i] + error)
```

#### 1.4. Compute Bias Term

```python
self.b = np.mean(y - (self.alpha * y) @ K)
```

#### 1.5. Prediction Function

```python
def predict(self, X):
    K = self.kernel_function(X, self.X_train)
    return np.sign((self.alpha * self.y_train) @ K.T + self.b)
```

Complete code is as follows:

```python
import numpy as np

class SVMClassifier:
    def __init__(self, C=1.0, kernel='linear', gamma=1.0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def linear_kernel(self, X, Y):
        return np.dot(X, Y.T)

    def rbf_kernel(self, X, Y):
        return np.exp(-self.gamma * np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=2) ** 2)

    def kernel_function(self, X, Y):
        if self.kernel == 'linear':
            return self.linear_kernel(X, Y)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X, Y)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.X_train = X
        self.y_train = y

        K = self.kernel_function(X, X)
        for _ in range(100):
            for i in range(n_samples):
                prediction = (self.alpha * y) @ K[:, i] + self.b
                error = y[i] * prediction - 1
                if error < 0:
                    self.alpha[i] = min(self.C, self.alpha[i] + error)

        self.b = np.mean(y - (self.alpha * y) @ K)

    def predict(self, X):
        K = self.kernel_function(X, self.X_train)
        return np.sign((self.alpha * self.y_train) @ K.T + self.b)
```

### 2. Regression SVM Implementation

The goal of regression SVM is to fit a function such that most data points lie within the $\epsilon$-insensitive zone. Below are the implementation steps:

#### 2.1. Kernel Functions and Initialization

```python
class SVR:
    def __init__(self, C=1.0, epsilon=0.1, kernel='linear', gamma=1.0):
        self.C = C  # Penalty coefficient
        self.epsilon = epsilon  # Tolerance zone
        self.kernel = kernel  # Kernel function
        self.gamma = gamma  # Parameter for RBF kernel

    def linear_kernel(self, X, Y):
        return np.dot(X, Y.T)

    def rbf_kernel(self, X, Y):
        return np.exp(-self.gamma * np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=2) ** 2)

    def kernel_function(self, X, Y):
        if self.kernel == 'linear':
            return self.linear_kernel(X, Y)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X, Y)
```

#### 2.2. Initialize Lagrange Multipliers and Kernel Matrix

```python
def fit(self, X, y):
    n_samples, n_features = X.shape
    self.alpha = np.zeros(n_samples)
    self.alpha_star = np.zeros(n_samples)
    self.b = 0
    self.X_train = X
    self.y_train = y
    K = self.kernel_function(X, X)
```

#### 2.3. Update Lagrange Multipliers

```python
for _ in range(100):
    for i in range(n_samples):
        prediction = (self.alpha - self.alpha_star) @ K[:, i] + self.b
        error = y[i] - prediction
        if abs(error) > self.epsilon:
            self.alpha[i] = min(max(self.alpha[i] + self.C * error, 0), self.C)
            self.alpha_star[i] = min(max(self.alpha_star[i] - self.C * error, 0), self.C)
```

#### 2.4. Compute Bias Term

```python
self.b = np.mean(y - (self.alpha - self.alpha_star) @ K)
```

#### 2.5. Prediction Function

```python
def predict(self, X):
    K = self.kernel_function(X, self.X_train)
    return (self.alpha - self.alpha_star) @ K.T + self.b
```

Complete code is as follows:

```python
import numpy as np

class SVR:
    def __init__(self, C=1.0, epsilon=0.1, kernel='linear', gamma=1.0):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma

    def linear_kernel(self, X, Y):
        return np.dot(X, Y.T)

    def rbf_kernel(self, X, Y):
        return np.exp(-self.gamma * np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=2) ** 2)

    def kernel_function(self, X, Y):
        if self.kernel == 'linear':
            return self.linear_kernel(X, Y)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X, Y)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.alpha_star = np.zeros(n_samples)
        self.b = 0
        self.X_train = X
        self.y_train = y

        K = self.kernel_function(X, X)
        for _ in range(100):
            for i in range(n_samples):
                prediction = (self.alpha - self.alpha_star) @ K[:, i] + self.b
                error = y[i] - prediction
                if abs(error) > self.epsilon:
                    self.alpha[i] = min(max(self.alpha[i] + self.C * error, 0), self.C)
                    self.alpha_star[i] = min(max(self.alpha_star[i] - self.C * error, 0), self.C)

        self.b = np.mean(y - (self.alpha - self.alpha_star) @ K)

    def predict(self, X):
        K = self.kernel_function(X, self.X_train)
        return (self.alpha - self.alpha_star) @ K.T + self.b
```

## Experimental Methods

### 1. Iris Dataset

The Iris dataset is a balanced dataset with 150 samples, each having four features across three classes. First, we visualize it to examine its linear separability. The visualization is as follows:

![iris](./asset/iris.png)

It can be seen that the data still has good linear separability, but there is some overlap near the boundaries (between versicolor and virginica). Therefore, we apply a soft margin SVC to handle this problem. For the classification strategy, we use 'one-vs-rest' (ovr), and evaluation metrics such as Precision, Recall, and F1-Score are employed. The results are discussed in the next section.

### 2. Ice-Cream Dataset

The Ice-Cream dataset is a regression task containing only one continuous feature. Visualizing the two variables' data gives:

<img src="./asset/icecream.png" alt="ice-cream" style="zoom:7%;" />

It can be seen that the data exhibits a strong linear characteristic. Therefore, a linear kernel SVC can be used for this task, with evaluation metrics including MSE (Mean Squared Error) and $R^2$. The specific results are discussed in the next section.

### 3. Wine-Quality Dataset

This is a univariate regression dataset with many features. We can plot the change of each feature against the dependent variable. The results show:

It is evident that the data's linear separability is not very good, so it may be necessary to use some non-linear kernels, such as the Gaussian kernel. The evaluation metrics are the same as those used for the Iris dataset.
