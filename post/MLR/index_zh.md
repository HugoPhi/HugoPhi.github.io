# 动手机器学习

English: [en](./index.html)

[TOC]

## 引言

​	在各类机器学习库日益成熟的趋势下，大家似乎更加青睐于使用封装好的学习器和算法库来处理工程和科学中的机器学习问题，但是从头实现一个机器学习算法仍然具有其必要性，尤其是对于一个想要精通机器学习初学者来说。对此观点，佐因有下：

- 手动实现算法的过程是完全可见且可调试的，这样更让人信服算法的效果。
- 对于出现的问题可以精准定位，同时对于一些问题有更加直观和深刻的体会（譬如：过拟合，权重初始化问题）。
- 理论结合实际。

据此，手动实现一些基本的机器学习算法还是很有必要的。而这个系列的文章将主要起到一个抛砖引玉的作用，引导大家如何去实现一个整洁的，可扩展的机器学习算法模型。

​	这篇文章还将把如何验证机器学习模型的各方面性能作为一个着重点。这是因为一个机器学习算法的好坏并不仅仅通过理论的证明就可以明晰，而是要通过大量的例子和高强度的验证来证明其性能，譬如在不同规模，场景的数据集上进行验证，并同之前的模型甚至人类进行对比。这主要是由于机器学习算法是算法之算法，我们评价一个学习器——尤其是一些比较复杂的学习器，更应该使用一些测评方法而不是证明方法，或者说完备的测试本来也是一种强有力的证明。这就好比我们通常看一个人会不会做饭是让其做几个拿首好菜菜看看效果，而不是找出一堆令人费解的符号和理论来证明它真的会做饭（当然这也是一个值得努力的方向）。当然这只是我个人理解，敬请斧正。

## 项目结构

​	本文实际上是一个对于项目的解说，这个项目包含比较完整的算法库，包管理，测试功能，至少用于学习通用途是绰绰有余的。下面我将讲解一下项目的整体结构：

### 1. 源码(src)

​	源码是指的对于机器学习模型的原始实现，我们希望我们写出来的代码可以被轻松地用到各个任务中去，那么我们就需要做好封装；而这些被封装好的模型自然需要被放到一个单独的文件夹下，这个文件夹就是src。另外，在src下我们也需要分类来存放各个算法使其不至于混乱。这里我采用的结构如下：

<details style="margin-left: 20px">
  <summary>监督学习</summary>
  <div style="margin-left: 20px">
    <li>逻辑回归</li>
    <details>
      <summary>决策树</summary>
      <div style="margin-left: 20px">
        <li>ID3</li>
        <li>C4.5</li>
        <li>CART</li>
      </div>
    </details>
    <details>
      <summary>支持向量机</summary>
      <div style="margin-left: 20px">
        <li>SVC</li>
      </div>
    </details>
    <details>
      <summary>神经网络</summary>
      神经网络算法描述
    </details>
  </div>
</details>
<details style="margin-left: 20px">
  <summary>无监督学习</summary>
  <div style="margin-left: 20px">
    <details>
      <summary>K 均值聚类</summary>
      K 均值聚类算法描述
    </details>
    <details>
      <summary>主成分分析</summary>
      主成分分析算法描述
    </details>
  </div>
</details>
如果你需要添加新的算法，只需要到对应的类别中去，创建一个平级的包，然后实现你的算法即可。

### 2. 测试(test)

​	正如前面提到过，我们会比较重视对于一个算法性能测试。这里我们采用了如下的结构：

<details style="margin-left: 20px">
  <summary>监督学习</summary>
  <div style="margin-left: 20px">
    <li>线性回归</li>
    <li>逻辑回归</li>
    <details>
      <summary>决策树</summary>
      <div style="margin-left: 20px">
        <li>watermelon2.0</li>
        <li>iris</li>
        <li>wine quality</li>
      </div>
    </details>
    <li>支持向量机</li>
    <li>神经网络</li>
  </div>
</details>
<details style="margin-left: 20px">
  <summary>无监督学习</summary>
  <div style="margin-left: 20px">
    <li>K 均值聚类</li>
    <li>主成分分析</li>
  </div>
</details>

如果你需要添加新的测试，只需要到对应的类别测试里面添加新的平级文件夹然后写你自己的测试即可，也可以根据你自己的想法随便写。



## 文章链接

​	文章的链接如下：

- 线性回归
- 逻辑回归
- 决策树
- [支持向量机](./post/svm/index_zh.html)
- 神经网络


## 知识储备

​	本文将会配合伪代码和python代码来讲解机器学习算法，所以请学习一些基本的python知识：譬如python面向对象，python错误处理等。同时为了更好的理解项目的组织方式你还需要学一下如何使用pip来组织一个python包，当然只要简单地学习一下就行了，别卷求你们了(bushi)。其它的内容比如高等数学和线性代数这些在本节并不是必要的，但这将会在另一个讲述机器学习原理系列的[文章](../MLT/index_zh.html)中用到。当然由于我是在linux上对这个项目进行开发的，所以学会一些基本的linux命令也是必要的。

​	最后，祝愿大家能够坚持学完这个系列，毕竟将理论付诸实践是一个漫长而艰辛的过程。

​	愿君多采撷，此物多有益。