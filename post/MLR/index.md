# ML From Scratch

中文： [zh](./index_zh.html)

[TOC]

## Introduction

​	As machine learning libraries continue to mature, many people seem to prefer using pre-packaged learners and algorithm libraries to handle machine learning problems in engineering and science. However, implementing a machine learning algorithm from scratch still has its value, especially for beginners who want to master the field. Here are a few reasons:

- The process of manually implementing an algorithm is fully visible and debuggable, making the algorithm's effectiveness more convincing.
- Problems can be pinpointed precisely, and some issues offer a more intuitive and profound understanding (e.g., overfitting, weight initialization issues).
- Theory combined with practice.

​	For these reasons, manually implementing some basic machine learning algorithms is necessary. This article series primarily serves as an introduction, guiding you on how to implement a clean, scalable machine learning algorithm model.

​	This article also focuses on how to validate the performance of machine learning models. The quality of a machine learning algorithm cannot be fully understood through theoretical proofs alone. It requires validation through many examples and rigorous testing—such as testing on datasets of varying sizes and scenarios—and comparison with previous models or even human performance. This is because machine learning algorithms are meta-algorithms, and evaluating a learner—especially more complex ones—should rely on evaluation methods rather than proof methods. In other words, thorough testing itself is a powerful form of validation. It's similar to how we judge someone's cooking ability by tasting their dishes, rather than proving their skills with confusing symbols or theories (though this is also a worthwhile pursuit). Of course, this is just my personal understanding; feel free to correct me.

## Project Structure

​	This article is essentially a walkthrough of the project, which contains a comprehensive algorithm library, package management, and testing functions—more than enough for learning purposes. Below, I’ll explain the overall structure:

### 1. Source Code (src)

​	The source code refers to the raw implementation of machine learning models. We hope that the code we write can be easily used in various tasks, so we need to ensure good encapsulation. These encapsulated models should naturally be placed in a separate folder, which is `src`. Additionally, we need to categorize the algorithms within `src` to prevent confusion. The structure I use is as follows:

<details style="margin-left: 20px">
  <summary>Supervised Learning</summary>
  <div style="margin-left: 20px">
    <li>Logistic Regression</li>
    <details>
      <summary>Decision Trees</summary>
      <div style="margin-left: 20px">
        <li>ID3</li>
        <li>C4.5</li>
        <li>CART</li>
      </div>
    </details>
    <details>
      <summary>Support Vector Machines</summary>
      <div style="margin-left: 20px">
        <li>SVC</li>
      </div>
    </details>
    <details>
      <summary>Neural Networks</summary>
      Description of Neural Network algorithms
    </details>
  </div>
</details>
<details style="margin-left: 20px">
  <summary>Unsupervised Learning</summary>
  <div style="margin-left: 20px">
    <details>
      <summary>K-means Clustering</summary>
      Description of K-means Clustering algorithm
    </details>
    <details>
      <summary>Principal Component Analysis</summary>
      Description of PCA algorithm
    </details>
  </div>
</details>

​	If you need to add a new algorithm, just create a new package in the corresponding category and implement your algorithm.

### 2. Tests (test)

​	As mentioned earlier, we place great emphasis on testing the performance of algorithms. We use the following structure for testing:

<details style="margin-left: 20px">
  <summary>Supervised Learning</summary>
  <div style="margin-left: 20px">
    <li>Linear Regression</li>
    <li>Logistic Regression</li>
    <details>
      <summary>Decision Trees</summary>
      <div style="margin-left: 20px">
        <li>Watermelon 2.0</li>
        <li>Iris</li>
        <li>Wine Quality</li>
      </div>
    </details>
    <li>Support Vector Machines</li>
    <li>Neural Networks</li>
  </div>
</details>
<details style="margin-left: 20px">
  <summary>Unsupervised Learning</summary>
  <div style="margin-left: 20px">
    <li>K-means Clustering</li>
    <li>Principal Component Analysis</li>
  </div>
</details>

​	If you need to add a new test, simply create a new folder at the same level within the corresponding category and write your test. You can also add tests as you wish based on your ideas.

## Article Links

The links to the articles are as follows:

- Linear Regression
- Logistic Regression
- Decision Trees
- [Support Vector Machines](./post/svm/index.html)
- Neural Networks

## Knowledge Prerequisites

​	This article will explain machine learning algorithms with pseudocode and Python code, so basic Python knowledge is required, such as object-oriented programming and error handling in Python. Additionally, for a better understanding of the project’s structure, you should learn how to use pip to manage a Python package—just a simple introduction is enough, no need to overdo it (just kidding). Other topics like advanced mathematics and linear algebra are not necessary for this section, but they will be useful in another series of articles on the principles of machine learning [here](../MLT/index_zh.html). Since I am developing this project on Linux, knowing some basic Linux commands is also necessary.

​	Finally, I wish you all the best in completing this series, as applying theory to practice is a long and challenging journey.

​	May you reap the rewards, as this knowledge will prove beneficial.