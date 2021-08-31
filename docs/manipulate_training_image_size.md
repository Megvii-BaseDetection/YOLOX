# Manipulating Your Training Image Size

This tutorial explains how to control your image size when training on your own data.

## 1. Introduction

There are 3 hyperparamters control the training size:

- self.input_size = (640, 640) &emsp; #(height, width)
- self.multiscale_range = 5
- self.random_size = (14, 26)

There is 1 hyperparameter constrols the testing size:

- self.test_size = (640, 640)

The self.input_size is suggested to set to the same value as self.test_size. By default, it is set to (640, 640) for most models and (416, 416) for yolox-tiny and yolox-nano.

## 2. Multi Scale Training

When training on your custom dataset, you can use multiscale training in 2 ways:

1. **【Default】Only specifying the self.input_size and leaving others unchanged.**

   If so, the actual multiscale sizes range from:

   [self.input_size[0] - self.multiscale_range\*32,  self.input_size[0] + self.multiscale_range\*32]

   For example, if you only set:

   ```python
   self.input_size = (640, 640)
   ```

   the actual multiscale range is [640 - 5*32, 640 + 5\*32], i.e., [480, 800].

   You can modify self.multiscale_range to change the multiscale range.

2. **Simultaneously specifying the self.input_size and self.random_size**

   ```python
   self.input_size = (416, 416)
   self.random_size = (10, 20)
   ```

   In this case, the actual multiscale range is [self.random_size[0]\*32, self.random_size[1]\*32], i.e., [320, 640]

   **Note: You must specify the self.input_size because it is used for initializing resize aug in dataset.**

## 3. Single Scale Training

If you want to train in a single scale. You need to specify the self.input_size and self.multiscale_range=0:

```python
self.input_size = (416, 416)
self.multiscale_range = 0
```

**DO NOT** set the self.random_size.
