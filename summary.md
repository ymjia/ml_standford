---
title: summary
header-includes:
  - \usepackage{cleveref}
author: jiayanming
---

## 目标函数Hypothesis
当遇到较复杂的问题时，我们希望通过已有的数据总结规律，进而利用此规律对新的数据作出判断。例如我们想对当前地区的房子进行估价，房子的价值与其一些固有属性有关，如面积，户型，通勤等等。即，对于一个向量 
$$m_i = [size, layout, commute, ...]$$

我们的目标是找到一个函数映射，由$$m_i$$映射到$$price_i = h_{theta}(m_i)$$

为此，我们假定一个目标函数形式，如：
$$h_{\Theta}(m_i) = \theta_0 + \theta_1 * m_i[1] + \theta_2 * m_i[2]$$

利用已有数据（训练集）对函数进行“训练”，获得能够最符合训练集的参数$$\Theta = [\theta_1, \theta_2, ..., \theta_n]$$，利用得到的$$\inline \Theta$$代入到目标函数 $$\inline h_{\Theta}$$对新数据进行预测。

## 损失函数Cost Function
有了目标函数后

## 梯度下降法 Gradient descent

## 线性回归问题(拟合问题) Linear Regression



## 逻辑回归问题(分类问题) Logistic Regression

## 神经网络 Neral Network

