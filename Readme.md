# Smart Amoeba

EN|[简中](Readme_CN.md)

A tinyML inference stack.

This document also contain a general introduction and overview of tinyML.

<!-- IMPORTANT Written in 3 am midnight. Don't be harsh alright? -->
<!-- TODO @ninn55 Check for spelling error and grammar error! -->

## Introduction

Moving AI (artificial intelligence) to embedded application can post certain challenges. Compared to cloud and mobile computing, the ecosystem for embedded hardware can be fragmented. System can vary from software to hardware stack, from build system to compiler, from ISA to peripherals. Embedded device at its core, is a highly customized computer system aiming at wide range of applications, which caused the highly customized software and hardware design. OS(Operating system) ranging from bare-metal to RTOS(Real time operating system), to micro-kernel, to Linux, is developed to fit onto different hardware. Power consumption can vary from micro-watts to hundreds of watts to accommodate from sensory analysis to self-driving cars and drones. A typical embedded system has certain traits:

<!-- TODO @BaconXu Changed mandarin content please verify-->

* Limited computing resources: AI/ML algorithm usually takes more computation. Running a programs needs CPU(Central Computing Unit) time and RAM(Randomly Accessed Memory). In terms, they are determined by limited power. This will cause significant challenges, when porting AI algorithm to edge.

* Fragmented ecosystem: The capacity of memory can range from KiloBytes to MegaBytes, from SRAM to SDRAM to DRAM. The Coremark:tm: score ranging from couple dozen to thousands. SIMD and float point operation is also limited to specific hardware platform. HAL(hardware abstraction layer) can also be different to fit application scenarios.

* Lack of Wildly accepted benchmarking: processors from different hardware vender have different debugging interface, simple OS or bare-metal. Code for user application and OS is usually mixed together. Along side with the lack of file system and shell interface makes it harder to run benchmark on a single application.

* Lack of portability: Software stack for commercial use is always business logic oriented. Portability is sacrificed for speed and space.

AI as a broad term means machine showing human intelligence. Recently, the most straight forward way is through neural network. But the NN itself is usually unexplainable and needed to be trained or verified on a large dataset. Designing an algorithm for embedded hardware can be especially challenging:

* Lack of mature use cases: Academically and commercially speaking most advances in deep learning is for cloud for mobile platform.
* Lack of open dataset: Most open source dataset as of 2021 is designed for cloud uses.
* Lack of mature model architecture: There is no widely accepted model architecture for embedded devices.
* Lack of efficient and portable model format: Currently the most used model format for TinyML is ONNX and tflite. But either of them is designed specifically for embedded device. ONNX is designed to be a common format between training framework. TfLite is designed to be run on mobile devices. ONNX is complex and hard to cover all operations in implementation. Decoding tflite model needs significant computing power.

The definition of TinyML given by tinyML organization is

> Tiny machine learning is broadly defined as a fast growing field of machine learning technologies and applications including hardware, algorithms and software capable of performing on-device sensor data analytics at extremely low power, typically in the mW range and below, and hence enabling a variety of always-on use-cases and targeting battery operated devices.

To the best of our knowledge, the definition of TinyML can be quantified as:

<!-- TODO @BaconXu please verify-->

* Always-on power consumption: < 1 mW
* RAM needed: < 100 KB
* ROM needed: < 1 MB
* Operation Count: < 10 MMAC / Inf
* Latency: < 100 mS / Inf

Designing a TinyML application will face challenge from all front. On one hand, AI algorithms take huge computing power. On another hand, embedded ecosystem is fragmented and embedded devices have limited computing power.

* Due to supply issue or advancing technology, the processor baring the TinyML workload need to be change regularly. When that inevitably happens, the inference solution needs to be ported as well.

* A necessary tool is needed to easy the process of deploying model from training framework. And make this process automated.

* When optimizing framework for 

### TinyML Inference Solution

#### Tensorflow Lite Micro

#### MicroTVM

<!-- TODO @BaconXu @ninn55 add later-->

#### GLOW

<!-- TODO @BaconXu @ninn55 add later-->

#### Tengine

<!-- TODO @BaconXu @ninn55 add later-->

## Our Solution

## Benchmark Result

### Latency Comparison

### MLPerf:tm: Tiny Inference

MLPerf:tm: Tiny is developed and published by MLCommons with help from community members.

> We (MLCommons) present MLPerf Tiny, the first industry-standard benchmark suite for ultra-low-power tiny machine learning systems.

With the recent advancement in TinyML technology, and the wildly fragmented ecosystem for embedded, a widely accepted benchmark suit is needed to quantify chip's AI compute capability, heterogeneous or otherwise. MLPerf:tm: Tiny is the first published benchmark suit specifically designed for TinyML. For the first round of MLPerf:tm: Tiny, there are four submitters, and they all use different hardwares to bare the load, a RISC-V micro-controller, an ASIC, a Cortex-A ARM processor, a FPGA, and the reference system is a Cortex-M micro-controller. This first submission perfectly demonstrated the fragmentation of embedded ecosystem and the challenge presented when designing a benchmark suit specifically for TinyML.

As a **proof of concept**, our team participated in this round of MLPerf Tiny where we demonstrated the AI compute capability and potential applications in AI for RISC-V micro-controllers.

![SCEPU01 Design-A](https://i.loli.net/2021/05/07/i2uDjlAMXxQEswC.png)

As a **submitter** and **reviewer**, we closely collaborated with parties involved including Harvard and EEMBC submit a RISC-V MCU entry. During this submission, we used a custom designed silicon solution implementing RV32I MAC instruction set with FPU to support single precision float point arithmetic. This SoC is designed by PCL and fabricated by SMIC (Semiconductor Manufacturing International Corporation) 55nm process and can operate with more than 300MHz clock speed. With 2MB SRAM and a plethora of interfaces, including SPI I2C UART GPIO and CAN bus.

![DSC04840-min.JPG](https://i.loli.net/2021/05/18/LuiaMmdzOs154fE.jpg)

There are four test cases in the test suit, covering most well established use cases for TinyML as of 2021, image classification, audio wake words, visual wake words, and anomaly detection. Anomaly detection uses deep auto-encoder to detect anomalies in audio. The neural network essentially took FFT encoded audio spectrum and out put another spectrum. By comparing the distance between the input and output, the ability of the network to encode and decode the input audio is quantified. Since the auto-encoder is trained only on normal audio data from a toy car, if the distance from input and output is larger than a given threshold, the neural network lack the ability to encode-decode the given audio spectrum. Then we assume the audio is from a abnormal toy car. The task image classification using the classic ResNet structure. By using the residual block to ease the training process. ResNet is a more de facto neural network structure for image classification task. The keyword spotting(audio wake word) and visual wake word both use DS-CNN network architecture, which uses a mobile net inspired network block. Audio wake word recognize specific keyword in a stream of audio data by classifying input spectrum and output an one hot encoded vector. And visual wake word took a preprocessed image and output one hot vector denoting person or not-a-person.

On the test suit side, a closed sourced benchmark runner is running on the host machine to generate a comparable and fare result for every submitter. The runner communicate with host machine with UART using a text based protocol then measure inference latency with device's internal timer. 

![MLPerf Tiny](bin\res\tinyperf.png)

Our team developed firmware for all 4 test case to make a more thorough comparison. We modified TensorFlow Lite Micro 2.3.1 and compiled on RISC-V GCC 10.2.0. All code/result for our system will be open source on GitHub after publication.

You can check out the published result from MLCommons. The paper and press coverage link will be updated after June 15, 2021.

#### Preliminary result

**Disclaimer**: The reference submission is developed by MLCommons. The RISC-V result is presented by us. But this result is still under embargo by MLCommons. The result presented here is only meant to be a early show-off, and for internal use only. All parties involved **DONOT** ensure the correctness of this results.

|  Submitter | Device  | Processor| Manufacturer & nodes | Software  | Latency VWW  | Latency IC  | Latency KWS  | Latency AD  |
|---|---|---|---|---|---|---|---|---|---|
| Harvard(Reference)  | Nucleo-L4R5ZI  |Arm Cortex-M4 w/ FPU   | TSMC 180nm  | Tensorflow Lite for Microcontrollers  |  603.14 |  704.23 | 181.92  |  10.40 |
|  Peng Cheng Laboratory |  PCL Scepu02 | RV32IMAC with FPU(1)  | SMIC 55nm  |  TensorFlowLite for Microcontrollers 2.3.1 (modified) | 846.74  |  1239.16 | 325.63  |  13.65 |

Latency data in the above table is in ms.

## Roadmap and Future



## Developers

|| Xu Xuesong  |  Niu Wenxu |
|---|---|---|
|Email| xuxs@pcl.ac.cn  |  wniu@connect.ust.hk |
|Github| https://github.com/coolbacon  |  https://github.com/ninn55 |

I you want to contribute or have any advice please contact either of us by email.

## Acknowledgements

Initially, This project is developed with support from Peng Cheng Laboratory and Jide OS Co.,Ltd.

|Peng Cheng Laboratory   |Jide OS   |
|---|---|
| [![pcl](http://www.szpclab.com/webimages/logo.png)](http://www.szpclab.com/)  | [![jideos](http://jideos.com/images/nav_logo.png)](http://jideos.com/)  |


## Reference

<!-- TODO @BaconXu Please fill all the references with proper name-->

* https://www.tinyml.org/
* https://arxiv.org/pdf/2010.08678.pdf
* https://arxiv.org/pdf/2003.04821.pdf
* 



