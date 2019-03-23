#https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l01c01_introduction_to_colab_and_python.ipynb#scrollTo=5QyOUhFw1OUX
import numpy as np

def hello(x, y):
    if x > y:
        print(x, "is greater than ", y)
    elif y > x:
        print(x, "is less than", y)
    else:
        print(x, "is equal to ", y)


hello(100, 45)

for i in range(8, 25, 5):
  print(i)
print("...............")

for i in range(5):
    print(i)
print("...............")

for i in [0,1,2,3,5]:
  print(i)
print("...............")

i = 0
while i < 2:
  print(i)
  i += 1
print("Numpy & List")

a = np.array(["Hello", "World"])
a = np.append(a, "!")# append ! to a
print("Current array: {}".format(a))

print("\nShowing some basic math on arrays")
b = np.array([0,1,4,3,2,4,4,7,8])
print("Max: {}".format(np.max(b)))
print("Average: {}".format(np.average(b)))
print("Max index: {}".format(np.argmax(b)))

print("\nUse numpy to create a [3,3] dimension array with random number")
c = np.random.rand(8, 3) # 8 array of 3 elements
print(c)