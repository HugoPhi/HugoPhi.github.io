# SVM

[TOC]

## 分类SVM算法理论

### 1. 基本SVM

#### 1.1. 问题描述

​	分类问题的核心在于找到一个能够有效分割不同类别样本的决策边界。在二维空间中，这个决策边界可以是一个线，三维空间中是一个平面，而在高维空间中则是一个超平面。对于线性可分的情况，假设数据集由两类标签组成，目标是找到一个最优超平面，将两类数据分开，并且尽可能增大两类样本到超平面的距离（即间隔）。这样不仅可以使分类更加准确，也增强了模型对噪声数据的鲁棒性。

#### 1.2. SVM的工作原理

​	SVM的核心思想是在数据空间中寻找一个能够最大化间隔的超平面。该超平面由一组特定的样本点（即支持向量）定义，这些支持向量是离超平面最近的样本点。SVM的目标是最大化超平面与支持向量之间的距离（即间隔），从而使分类模型更具有泛化性。

假设数据集包含$ N$ 个样本点 $(x_i, y_i)$，其中 $ x_i \in \mathbb{R}^d$ 表示第 $ i$ 个样本的特征向量，$ y_i \in \{-1, 1\}$ 表示样本的标签。SVM的目标是寻找一个线性决策函数：

$$
f(x) = w \cdot x + b
$$

​	其中，$ w $ 是权重向量，$ b $ 是偏置项，使得函数 $ f(x) = 0 $ 对应于分割超平面。对于满足线性可分的数据，SVM希望找到最优的$ w $ 和 $ b$，使得不同类别的样本点离分割超平面的间隔最大。

#### 1.3. 优化目标

​	要实现间隔最大化，SVM构建的优化目标如下：

$$
\text{maximize} \quad M = \frac{2}{\|w\|}
$$

​	即最大化超平面到支持向量的间隔$ M$。通过适当的变换，SVM的优化问题可以被表示为一个约束条件下的二次优化问题：

$$
\min \frac{1}{2} \|w\|^2
$$

$$
\text{s.t.} \quad y_i (w \cdot x_i + b) \geq 1, \quad i = 1, 2, \ldots, N
$$

上式中，约束条件$ y_i (w \cdot x_i + b) \geq 1$ 表示所有样本点的类别在超平面的约束下得到正确分类，且距离不小于1。

#### 1.4. SVM求解方法

​	SVM的求解方法通常使用拉格朗日乘子法和KKT（Karush-Kuhn-Tucker）条件，将约束优化问题转化为无约束优化问题。引入拉格朗日乘子$ \alpha_i$ 后，目标函数变为：

$$
L(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^{N} \alpha_i [y_i (w \cdot x_i + b) - 1]
$$

通过对$ L(w, b, \alpha) $ 关于 $ w $ 和 $ b$ 求偏导并令其为0，可以得到对偶问题。最终的对偶优化目标为：

$$
\max \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
$$

$$
\text{s.t.} \quad \sum_{i=1}^{N} \alpha_i y_i = 0, \quad \alpha_i \geq 0, \quad i = 1, 2, \ldots, N
$$

优化求解得到 $\alpha$ 后，权重向量$ w$ 和偏置 $ b$ 可以被计算出来：

$$
w = \sum_{i=1}^{N} \alpha_i y_i x_i
$$

然后选取一个支持向量$x_k $计算偏置：

$$
b = y_k - w \cdot x_k
$$

一旦得到 $w $ 和 $b $，我们就可以通过分类决策函数$ f(x) = \text{sign}(w \cdot x + b)$对新的数据点进行分类。

#### 1.5. 总结

​	基本SVM的优化目标是找到一个最大化类别间隔的超平面，从而提高模型的鲁棒性和泛化能力。通过拉格朗日对偶问题的求解，SVM能够在训练过程中自动选择最有影响力的样本点（支持向量），最终得到一个分类超平面。

### 2. 软间隔SVM

​	在实际问题中，数据往往不是完全线性可分的，可能存在噪声点或重叠区域。为了解决这一问题，SVM引入了软间隔（Soft Margin）概念。软间隔SVM允许在分类边界周围存在一定的误分类样本点，从而平衡分类精度和模型的泛化能力。通过添加一个松弛变量$\xi $，软间隔SVM的目标函数可以表示为：

$$
\min \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i
$$

​	其中$C $ 是惩罚参数，用于控制误分类的容忍度。较大的 $ C $值意味着模型更重视分类准确性，尽量减少误分类，但可能导致过拟合；较小的 $ C $ 值则更倾向于增大间隔，允许更多误分类，从而增强模型的泛化能力。

### 3. 核技巧

​	对于非线性可分的数据，SVM使用核技巧（Kernel Trick）来将数据映射到高维空间，以实现线性可分。在高维空间中，SVM可以通过线性分离方法对复杂的非线性数据进行分类。常用的核函数包括线性核、多项式核、径向基函数（RBF）核和 Sigmoid 核。

​	核函数的引入允许 SVM 将非线性问题转化为线性问题，从而极大地扩展了 SVM 的应用范围。常用的核函数形式为：

$$
K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)
$$

​	其中 $ \phi(x) $ 是将原始特征空间映射到高维空间的映射函数。核技巧不需要显式计算高维映射，只需通过核函数直接计算特征之间的相似度，因此计算效率较高。

核函数的选择和参数的设置将直接影响SVM分类模型的表现，需要根据具体问题的分布特点来调整。

​	在支持向量机中，核函数的主要作用是将数据从低维空间映射到高维空间，从而使非线性可分的数据在高维空间中变得线性可分。核函数的选择对于模型的性能有重要影响，不同的核函数适用于不同的数据分布和特征。以下是几种常见的核函数及其适用场景：

#### 3.1. 线性核（Linear Kernel）

**表达式**：$ K(x_i, x_j) = x_i \cdot x_j $

**适用场景**： 
线性核是最简单的核函数，适用于线性可分的数据集。在低维空间或者特征数远大于样本数的场景下，线性核表现良好。例如，文本分类、图像分类等高维稀疏特征数据通常适合使用线性核。在这些应用中，数据的类别边界往往接近线性分布，因而线性核能够有效且高效地进行分类。

**优缺点**：  

- **优点**：计算效率高，尤其在高维稀疏数据上表现出色。
- **缺点**：无法处理非线性数据。

#### 3.2. 多项式核（Polynomial Kernel）

**表达式**：$ K(x_i, x_j) = (x_i \cdot x_j + c)^d $

其中，$ c $ 是常数项，$ d $ 是多项式的阶数。

**适用场景**： 
多项式核适用于那些具有复杂交互关系的数据，但数据的非线性不明显。通过调整多项式的阶数 $ d $ 和常数项 $ c $，多项式核可以在较低维度上处理具有一定非线性的分类问题。它常用于图像处理和自然语言处理中，例如词向量间复杂关系的建模。

**优缺点**：  

- **优点**：适合中等非线性数据，能够通过调节阶数灵活处理不同复杂度的数据。
- **缺点**：在高维度和大规模数据集上计算成本较高，容易导致模型过拟合。

#### 3.3. 径向基核（RBF核）或高斯核（Gaussian Kernel）

**表达式**：$ K(x_i, x_j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right) $

其中，$ \sigma $ 是用于调节分布范围的参数。

**适用场景**： 
RBF核是最常用的核函数，适用于大部分非线性分类问题，尤其在特征空间较为复杂的场景中表现出色。它具有局部化特性，对相似性较高的数据点具有较强的响应。RBF核常用于生物信息学、图像识别和手写数字识别等需要捕捉复杂边界的领域。

**优缺点**：  

- **优点**：能够灵活地处理高度非线性的分类任务，具有较强的模型泛化能力。
- **缺点**：对参数 $ \sigma $ 敏感，参数设置不当容易导致过拟合或欠拟合。

#### 3.4. Sigmoid核

**表达式**：$ K(x_i, x_j) = \tanh(\alpha x_i \cdot x_j + c) $

其中，$ \alpha $ 和 $ c $ 为常数，$ \tanh $ 是双曲正切函数。

**适用场景**： 
Sigmoid核在某些方面类似于神经网络的激活函数，适用于具有神经网络特性的分类问题。它在二类分类任务中使用较多，适合小规模、非线性不特别显著的分类任务，且数据分布较规则。Sigmoid核可用于识别二类模式或特定分类问题的初步实验，但其表现通常不如RBF核或多项式核。

**优缺点**：  

- **优点**：适用于二类分类任务，特别是早期的神经网络模型中。
- **缺点**：不一定满足所有核函数的Mercer定理，因此在特定场景下可能无法收敛，效果不稳定。

#### 3.5. 拉普拉斯核（Laplacian Kernel）

**表达式**：$ K(x_i, x_j) = \exp\left(-\frac{\|x_i - x_j\|}{\sigma}\right) $

**适用场景**： 
拉普拉斯核与RBF核相似，但它使用L1距离而不是L2距离，适用于一些具有局部相似性且数据噪声较多的场景。其在信号处理、图像分割等需要对局部特征敏感的应用中更为常见。

**优缺点**：  

- **优点**：对异常值鲁棒性更强，适合噪声较多的数据。

- **缺点**：计算效率可能较低，适用于特定的局部特征任务。

​	核函数的选择需要根据数据的分布情况和问题的特性进行调整。在实际应用中，可从简单的核函数（如线性核）开始，如果发现模型表现不佳，则尝试更为复杂的核（如RBF核、多项式核），并结合交叉验证来优化核函数的参数。


## 回归SVM算法理论

### 1. 基本回归SVM

#### 1.1 问题描述

回归问题的目标是通过学习训练数据中的模式，预测连续的数值输出。在支持向量回归（Support Vector Regression, SVR）中，模型拟合的目标是找到一个函数，使得绝大多数数据点的预测误差不超过指定的容忍范围 $ \epsilon $。与分类SVM不同，SVR不再关注将数据分为不同类别，而是构建一个容忍误差的“间隔管道”，使绝大部分样本点都位于此管道内，并通过优化使模型对噪声和异常点的影响最小化。

在数学表示上，给定数据集 $ (x_i, y_i) $，SVR尝试找到一个线性函数：

$$
f(x) = w \cdot x + b
$$

使得对于绝大部分样本点 $ (x_i, y_i) $，预测值 $ f(x_i) $ 和真实值 $ y_i $ 之间的差值不超过容忍范围 $ \epsilon $。这意味着模型允许一定程度的误差，但超出该容忍区间的误差会被惩罚。

#### 1.2 SVR的工作原理

SVR的工作原理核心在于构建一个 $ \epsilon $-不敏感区间（epsilon-insensitive zone），即一个允许一定误差的间隔。这个间隔称为“间隔管道”或“回归带”。在该区间内，预测误差被忽略（即不计算损失），而超出此范围的误差则会受到惩罚。

优化问题的目标是最小化模型的复杂度（通过控制 $ w $ 的大小），并将大部分样本点包含在 $ \epsilon $-不敏感区间内。具体地，优化问题的表示为：

$$
\min \frac{1}{2} \|w\|^2
$$

$$
\text{s.t.} \quad |y_i - (w \cdot x_i + b)| \leq \epsilon
$$

该约束条件表明所有数据点都尽量位于误差小于 $ \epsilon $ 的区间内。为了进一步处理那些超出容忍区间的样本点，SVR引入了松弛变量 $ \xi $ 和 $ \xi^* $ 来表示正、负方向上的偏差：

$$
\text{s.t.} \quad y_i - (w \cdot x_i + b) \leq \epsilon + \xi_i
$$

$$
(w \cdot x_i + b) - y_i \leq \epsilon + \xi_i^*
$$

最终，SVR的优化目标变为：

$$
\min \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)
$$

其中 $ C $ 为惩罚参数，控制模型对超出 $ \epsilon $-不敏感区间的误差的容忍度。较大的 $ C $ 值使模型更倾向于减少误差，但可能导致对噪声的过度拟合；较小的 $ C $ 值则允许更多误差，提升模型的泛化能力。

#### 1.3 SVR求解方法

SVR求解的核心是通过拉格朗日对偶方法将约束优化问题转化为无约束优化问题。通过引入拉格朗日乘子 $ \alpha $ 和 $ \alpha^* $，可以构建对偶问题，从而简化计算。优化过程的最终结果是通过支持向量计算出权重向量 $ w $ 和偏置 $ b $，得到的回归模型可用于预测新的样本点。

### 2. 核技巧在回归中的应用

在实际应用中，许多数据集并非线性可分，即数据与输出之间的关系不是简单的线性关系。为了解决这一问题，SVR可以使用核函数（Kernel Function）将数据映射到高维空间，使得在该高维空间中回归问题变得接近线性可分。这种通过核函数映射实现的高效运算方式，避免了直接计算高维空间坐标，从而减少计算复杂度。

常见的核函数包括：

- **线性核**：适用于数据线性相关性较强的情况。
- **多项式核**：适用于具有复杂特征交互的数据。
- **径向基核（RBF）**：适用于大多数非线性问题，能很好地处理局部相似性。
- **Sigmoid核**：在小规模数据的二类分类任务中使用较多。

核函数的选择直接影响模型的表现，需要结合数据的特点和具体任务进行选择与调优。径向基核（RBF）通常是默认的选择，因为其非线性特性适合多数实际应用。

### 3. 参数选择和调优

SVR的主要参数包括：

- **惩罚参数 $ C $**：控制模型对超出容忍范围样本的惩罚力度。较大的 $ C $ 值会使模型更关注误差，趋向于较低的训练误差，但可能导致过拟合。较小的 $ C $ 值则允许更多误差，从而增强模型的泛化能力。

- **容忍区间宽度 $ \epsilon $**：决定了模型在何种误差范围内不计算损失。适当增大 $ \epsilon $ 可以减少对噪声数据的敏感性，从而提高模型的稳定性。

- **核函数参数（如RBF核的 $ \gamma $**）：控制核函数的特征空间映射，影响模型的非线性拟合能力。

在实际使用中，这些参数通常需要通过交叉验证来选择，找到最优组合以获得最佳的回归效果。



## 手动实现SVM算法

​	在本节中，我们将分步骤手动实现分类SVM和回归SVM算法，分别针对分类任务和回归任务的需求进行代码编写。实现中不使用任何机器学习库，仅依靠基础数值计算库来手动构建算法流程，帮助理解SVM算法的核心原理和计算过程。

### 分类SVM实现

​	分类SVM的目标是找到一个最佳分隔超平面，以最大化间隔的方式将不同类别的样本分开。以下为实现步骤及代码：

#### 1. 初始化参数和核函数

```python
import numpy as np

class SVMClassifier:
    def __init__(self, C=1.0, kernel='linear', gamma=1.0):
        self.C = C  # 惩罚系数
        self.kernel = kernel  # 核函数类型
        self.gamma = gamma  # RBF核的参数

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

#### 2. 计算核矩阵并初始化拉格朗日乘子

```python
def fit(self, X, y):
    n_samples, n_features = X.shape
    self.alpha = np.zeros(n_samples)
    self.b = 0
    self.X_train = X
    self.y_train = y

    # 计算核矩阵
    K = self.kernel_function(X, X)
```

#### 3. 优化拉格朗日乘子（SMO简化实现）

```python
for _ in range(100):  # 设置迭代次数
    for i in range(n_samples):
        # 计算预测值
        prediction = (self.alpha * y) @ K[:, i] + self.b
        # 更新 alpha_i 的值
        error = y[i] * prediction - 1
        if error < 0:
            self.alpha[i] = min(self.C, self.alpha[i] + error)
```

#### 4. 计算偏置项

```python
self.b = np.mean(y - (self.alpha * y) @ K)
```

#### 5. 预测函数

```python
def predict(self, X):
    K = self.kernel_function(X, self.X_train)
    return np.sign((self.alpha * self.y_train) @ K.T + self.b)
```

完整代码如下：

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

### 回归SVM实现

回归SVM的目标是拟合一个函数，以使绝大多数数据点在 \( \epsilon \) 不敏感区间内。以下是实现步骤：

#### 1. 核函数和初始化

```python
class SVR:
    def __init__(self, C=1.0, epsilon=0.1, kernel='linear', gamma=1.0):
        self.C = C  # 惩罚系数
        self.epsilon = epsilon  # 不敏感区间
        self.kernel = kernel  # 核函数
        self.gamma = gamma  # RBF核参数

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

#### 2. 初始化拉格朗日乘子和核矩阵

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

#### 3. 更新拉格朗日乘子

```python
for _ in range(100):
    for i in range(n_samples):
        prediction = (self.alpha - self.alpha_star) @ K[:, i] + self.b
        error = y[i] - prediction
        if abs(error) > self.epsilon:
            self.alpha[i] = min(max(self.alpha[i] + self.C * error, 0), self.C)
            self.alpha_star[i] = min(max(self.alpha_star[i] - self.C * error, 0), self.C)
```

#### 4. 计算偏置项

```python
self.b = np.mean(y - (self.alpha - self.alpha_star) @ K)
```

#### 5. 预测函数

```python
def predict(self, X):
    K = self.kernel_function(X, self.X_train)
    return (self.alpha - self.alpha_star) @ K.T + self.b
```

完整代码如下：

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





## 实验方法

### iris数据集

​	iris数据集是一个有150条4种特征的3类别平衡数据集。首先我们对其进行可视化以考察其线性可分性。可视化得到：

![iris](./asset/iris.png)

​	可以看到这里的数据还是具有很好的线性可分性的，但是在边界处具有一些交叉（versicolor和virginica），因此我们适用软间隔SVC来处理这个问题，对于分类策略使用'ovr'，评判指标采用Precision，Recall，F1-Score等。结果在下一节阐述。



### ice-cream

​	ice-cream数据集是只包含一个连续特征的回归任务。可视化两个变量的数据得到：

![ice-cream](./asset/icecream.png)

可以看到数据呈现出很强的线性特性。则可以用线性核的SVC来做这个任务，评判指标采用MSE以及$R^2$。具体结果在下一节阐述。

### wine-quality

​	这是一个有许多特征的单变量回归数据集。我们可以通过对每一个特征对因变量的变化作图。得到：


可见数据的线性可分性并不是很好，所以可能需要适用一些非线性核，比如高斯核。评价指标和iris一样。



