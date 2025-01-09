print("Hello World")
inputs = [1 , 2, 3, 2.5]
weights1 = [0.2 , 0.8 , -0.5 , 1.0]
weights2 = [0.5 , -0.91 , 0.26 , -0.5]
weights3= [-0.26 ,-0.27 , 0.17 ,0.87 ]
a = len(inputs)
bias1 = 2.0
bias2 = 3.0
bias3 = 0.5
output1 = 0
output2 = 0
output3 = 0
for i in range (0,a):
    output1 += inputs[i] * weights1[i];
    output2 += inputs[i] * weights2[i];
    output3 += inputs[i] * weights3[i];

print(output1 + bias1)
print(output2 + bias2)
print(output3 + bias3)

