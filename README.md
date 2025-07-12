# 2025 Summer Research Project

> This folder record the 2025 summer research project

## Introduction

1. Device Management

The design is divided into two categories:
	(1)	Tensor-generating classes: Classes that generate tensors internally, or contain attributes that do so.
	(2)	Tensor-processing classes: Classes that operate solely on input tensors without generating new ones.

For category (1), all classes inherit from nn.Module, enabling consistent device management via .to(device).

For category (2), classes do not inherit from nn.Module, but ensure that all outputs reside on the same device as the inputs (e.g., output.to(input.device)).

Only category-(1) components are permitted to generate tensors. This guarantees that any module combining generation and processing maintains internal buffers and inputs on the same device.
