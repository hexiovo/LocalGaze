Introduction
============

本项目是Hexi用于实现个人眼动追踪程序的包，如有问题请见谅

email:py_edu_mail\@163.com

Thanks
======

使用本包如遇问题请与我本人联系，如果有科研合作也请联络本人

我需要相关参数（所存储的model_info.xlsx文件），请好心人上传给我，以供我后续调节

如果有相关BUG请上传对应的Log

Statement
=========

本项目基于eyetrax：

[ck-zhang/EyeTrax: EyeTrax – webcam-based eye tracking made
simple](https://github.com/ck-zhang/EyeTrax)

所做改动：更改了九点验证的结果，使其可以实时显示当前眼动区域，并输出误差

增加了在九点验证基础上的多轮调试，使得结果处于可接受范围，直到接受才可以进入正式采集

且在这个过程中全程保持摄像头开启，增加了模型和数据存储功能

MainProgram
===========

Eyetracking:
------------

LocalGaze

Misc:
-----

眼动ROI绘制：ROIDrwaing

DadaAnalysis
------------

眼动数据处理：EyeEvaluate

 

Process
=======

LocalGaze
---------

0.环境准备

0.1.下载conda(miniconda也可)官网

[Download Success \| Anaconda](https://www.anaconda.com/download/success)

0.2.安装虚拟环境

点击env_requirement中的SetForLocalGaze.bat文件等待安装成功

0.3.设置参数

进入global_data中，以文本形式打开，摄像头数量代表选取的摄像头，0为默认

屏幕宽度和高度请改为相对应的分辨率

0.4.注：

LocalGaze的环境：python == 3.10

1.使用

双击LocalGaze.exe

读取信息点击确定，选择是否使用历史模型（建议按照被试信息存储）

如果选择历史模型，就选择对应.pkl即可

如果不，进入九点矫正；根据结果选择是否进行额外矫正

选择矫正则输入相关参数，然后进入正式实验，输入存储名称（可不同于历史模型，用于表征试次）

选择是否开启预测点显示

点击退出即可完成任务

2.数据存储

模型默认存储在Model_history中

结果保存在Data中

3.备注

更具体的指导语位于使用指导语及流程

ROIDrawing
----------

0.环境配置

0.1.下载conda(miniconda也可)官网

[Download Success \| Anaconda](https://www.anaconda.com/download/success)

0.2.安装虚拟环境

点击env_requirement中的SetForROIDrawing.bat文件等待安装成功

0.3.设置参数

进入global_data中，设置分辨率以及灰色屏幕的透明程度

1.使用

点击ROIDrawing.exe

在左上角显示对应的图形，点击后进行绘制，右键绘制，左键拖拽

完成后保存

2.数据存储

ROIDrawing\\ROIdata下

EyeEvaluate
===========

0.环境配置

0.1.下载conda(miniconda也可)官网

[Download Success \| Anaconda]

0.2.安装虚拟环境

 

Log
===

V1.1.0

更改了函数参数选择功能，可以进入Global_data中设置，其中如果要使用新的模型及其参数，请自由探索。

V1.2.0

更改了误差计算方案以及误差存储方案，尽可能多的获取信息。

V1.3.0

增加了瞳孔直径的计算方案

V1.4.0

更新了已知BUG

V1.5.0

更新环境处理能力，包装为exe

V1.5.1

fixed bugs

V1.5.2

killing bugs

V1.5.4

修正了额外校准阶段点显示问题

增加了摄像头选择功能

更新了环境处理中可能存在的BUG

V1.5.5

修正了可能存在的报错（tk线程问题）

增加了实时保存，将保存在/LocalGaze/Data/以txt形式保存，以防数据丢失

V1.5.6

增加了使用指导语及流程，方便进行实验操作

V1.6.0

增加了log功能

V1.7.0

更新了ROIDrawing部分

V1.7.2

修正BUG，增加了ROIDrawing的日志功能

V1.7.3

修正BUG，使得结果正确

V1.8.0

眼动数据处理程序EyeEvaluate正在processing

V1.8.1

增加了更完善的日志功能并封装

V1.8.2

添加了更严密的弹窗个性化，增加了文件与文件夹选择功能

V1.8.3

完成了ROI提取函数

增加了针对列的ROI识别功能

V1.8.4

修正了流程，增添了log的important功能
