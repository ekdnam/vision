img = imread("resources\sample_bw.jfif")
J = integralImage(img)
imwrite(J, 'output.jpeg')