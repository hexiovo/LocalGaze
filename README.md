本项目是和Hexi用于实现个人眼动追踪程序的包，如有问题请见谅

email:py_edu_mail\@163.com

使用本包如遇问题请与我本人联系，如果有科研合作也请联络本人

我需要相关参数（所存储的model_info.xlsx文件），请好心人上传给我，以供我后续调节

 

本项目基于eyetrax：

[ck-zhang/EyeTrax: EyeTrax – webcam-based eye tracking made
simple](https://github.com/ck-zhang/EyeTrax)

所做改动：更改了九点验证的结果，使其可以实时显示当前眼动区域，并输出误差

增加了在九点验证基础上的多轮调试，使得结果处于可接受范围，直到接受才可以进入正式采集

且在这个过程中全程保持摄像头开启

 

增加了模型和数据存储功能

 

 

使用流程：

0.环境准备

0.1下载conda(miniconda也可)官网

[Download Success \| Anaconda](https://www.anaconda.com/download/success)

0.2安装虚拟环境，点击env_requirement中的SetForLocalGaze.bat文件等待安装成功

 

1.使用

双击LocalGaze.exe

读取信息点击确定，选择是否使用历史模型（建议按照被试信息存储）

如果选择历史模型，就选择对应.pkl即可

如果不，进入九点矫正；根据结果选择是否进行额外矫正

选择矫正则输入相关参数，然后进入正式实验，输入存储名称（可不同于历史模型，用于表征试次）

选择是否开启预测点显示

点击退出即可完成任务

 

python == 3.10

 

 

模型默认存储在Model_history中

结果保存在Data中

 

Log：

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
