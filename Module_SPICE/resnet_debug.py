from resnet import ResNet
import time








r_obj = ResNet()


ticp = time.time()
for i in range(1000):
    r_obj.run_model([1,2,3])
tocp = time.time()


o_time = tocp - ticp
print("obj time:", o_time)





































#

# fin
