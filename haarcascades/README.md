# Haarcascades

## What are Haar features?

[This article](https://towardsdatascience.com/whats-the-difference-between-haar-feature-classifiers-and-convolutional-neural-networks-ce6828343aeb) summarizes the concept best
> A Haar-Feature is just like a kernel in CNN, except that in a CNN, the values of the kernel are determined by training, while a Haar-Feature is manually determined.

There are in all **four** types of Haar-features.

![haar-features](../assets/haarcascades/haar-features.png)

- The first two features are used to detect edges.
- The third feature is used to detect lines.
- The fourth feature detects slants, called a rectangle feature.

The kernels can look something like this

- First feature
    |_-1_|_-1_|**5**|
    |-|--|-|
    |_-1_|_-1_|**5**|
    |_-1_|_-1_|**5**|

- Second feature
    |**5**|**5**|**5**|
    |-|--|-|
    |_-1_|_-1_|_-1_|
    |_-1_|_-1_|_-1_|

- Third feature
    |_-1_|**5**|_-1_|
    |-|--|-|
    |_-1_|**5**|_-1_|
    |_-1_|**5**|_-1_|

- Fourth feature
    |**5**|_-1_|_-1_|
    |-|--|-|
    |_-1_|**5**|_-1_|
    |_-1_|_-1_|**5**|

As we do in CNNs, this 3x3 kernel moves across the image, performing some matrix multiplication operations, emphasizing some features and smoothing others. [[1]](#reference1)

Haar features specialize in extracting edges, lines, slants in an image. This makes them extremely useful in face detection. It can detect on our faces, with sufficiently good lighting, our eyes, nose, and the face boundary.

### Pros

- Don't have to train the entire Haar kernel every time, can retrain the weights on a small dataset as well.

### Cons

- Since the weights are manually determined, and have the special ability of edge and line detection, there may be some error while performing the task on a subject which does not have clear lines and edges.
  - For example, if the lighting is low, or the subject wears sunglasses, the haar features may not be sufficient enough to detect defining features.

## References

### Haar features

1. <a name = "reference1"></a>[What's the difference between Haar-feature classifiers and Convolutional Neural Networks?](https://towardsdatascience.com/whats-the-difference-between-haar-feature-classifiers-and-convolutional-neural-networks-ce6828343aeb)
1. [What are Haar features used in Face Detection?](https://medium.com/analytics-vidhya/what-is-haar-features-used-in-face-detection-a7e531c8332b)