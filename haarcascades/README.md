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

As we do in CNNs, this 3x3 kernel moves across the image, performing some matrix multiplication operations, emphasizing some features and smoothing others. <sup>[[1]](#hf1)

Haar features specialize in extracting edges, lines, slants in an image. This makes them extremely useful in face detection. It can detect on our faces, with sufficiently good lighting, our eyes, nose, and the face boundary.

### Pros

- Don't have to train the entire Haar kernel every time, can retrain the weights on a small dataset as well.

### Cons

- Since the weights are manually determined, and have the special ability of edge and line detection, there may be some error while performing the task on a subject which does not have clear lines and edges.
  - For example, if the lighting is low, or the subject wears sunglasses, the haar features may not be sufficient enough to detect defining features.

## Integral Image (Summed Area Table)

The concept is not as difficult as it sounds.

It is an effective way of calculating the sum of pixel values in a given image - or a rectangular subset of a grid (the given image). <sup>[[2]](#ii1).

This is then used for calculating the average intensity within the grid. It is preferred that the image is grayscale first.

### The algorithm

![Equation - initial](../assets/haarcascades/ii-eq-1.svg)

What it simply says, is, <b>the value at any point (x, y) in the summed-area table is the sum of all the pixels above and to the left of (x, y), inclusive.</b> <sup>[[3]](#ii3)

Time complexity-wise, this can be done easily.

![Equation - 2](../assets/haarcascades/ii-eq-2.svg)

We can simply get the value of a pixel in an II by adding the value of the pixel, the II value of the pixel to its left, the II value of the pixel above it, and subtracting the II value of the pixel which is directly top-left of it.

For simplicity reasons, we begin to calculate the II values from the top-left corner of the image.

Instead of just the pixels, we can also add the II values of smaller II's inside an image. Sounds a bit complicated, right? [This link](#ii1) can explain it to you in an easier way.

The recurrence relation

<i><var>s(x, y) = s(x, y - 1) + i(x, y)</var></i>

<var><i>ii(x, y) = ii(x - 1) + s(x, y)</i></var>

(sort of the same equation as above), is used to compute the II in one-pas over the image.<sup>[[4]](#ii4)</sup> Time-complexity is <var>O(1)</var>.

## References

### The Viola-Jones Paper

1. <a name = "viola-jones"></a>[Rapid Object Detection using a Boosted Cascade of Simple
Features](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)

### Haar features

1. <a name = "hf1"></a>[What's the difference between Haar-feature classifiers and Convolutional Neural Networks?](https://towardsdatascience.com/whats-the-difference-between-haar-feature-classifiers-and-convolutional-neural-networks-ce6828343aeb)
2. <a name = "hf2"></a>[What are Haar features used in Face Detection?](https://medium.com/analytics-vidhya/what-is-haar-features-used-in-face-detection-a7e531c8332b)

### Integral Image

1. <a name = "ii1"></a>[Computer Vision - The Integral Image](https://computersciencesource.wordpress.com/2010/09/03/computer-vision-the-integral-image/)
2. <a name = "ii2"></a>[integralImage](https://in.mathworks.com/help/images/ref/integralimage.html)
3. <a name = "ii3"></a>[Summed Area Table](https://en.wikipedia.org/wiki/Summed-area_table)
4. <a name = "ii4"></a>[Integral image-based representations](http://www.cse.yorku.ca/~kosta/CompVis_Notes/integral_representations.pdf)
