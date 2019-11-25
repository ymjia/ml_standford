---
title: summary
header-includes:
  - \usepackage{cleveref}
author: jiayanming
---

## 目标函数Hypothesis
当遇到较复杂的问题时，我们希望通过已有的数据总结规律，进而利用此规律对新的数据作出判断。例如我们想对当前地区的房子进行估价，房子的价值与其一些固有属性有关，如面积，户型，通勤等等。即，对于一个向量 
$m_i = [size, layout, commute, ...]$

我们的目标是找到一个函数映射，由$m_i$映射到$price_i = h_{theta}(m_i)$

N为此，我们假定一个目标函数形式，如：
	$h_{\Theta}(m^{(i)}) = \theta_0 + \theta_1 * m^{(i)}_1 + \theta_2 * (m^{(i)}_2)^2 + ...$
其中，上标(i)是训练集元素的id，下标是训练集元素向量中的元素id。

利用已有数据（训练集）对函数进行“训练”，训练目标是：获得一组参数$\Theta = [\theta_1, \theta_2, ..., \theta_n]$，使目标函数的输出结果尽可能“符合”训练集，然后可利用得到的$\Theta$代入到目标函数 $h_{\Theta}$对新数据进行预测。

## 损失函数Cost Function
有了目标函数后，为了找到最优的参数，我们需要对目标函数与训练集是否“符合”进行评估，为此引入损失函数$J(\Theta)$，此函数训练集上关于$\Theta$的函数，用来表征目标函数$h_{\Theta}(m_i)$的性能。对于训练集中的每个训练数据，利用损失函数评估当前参数下目标函数的性能。我们训练参数的目的可以更明确的表示为：对损失函数做优化，寻找令损失函数取得最小值的$\Theta$。

## 梯度下降法 Gradient descent


## 线性回归问题(拟合问题) Linear Regression

## 逻辑回归问题(分类问题) Logistic Regression

## 神经网络 Neral Network

