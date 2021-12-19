# ntut-talking-head
### 1.Notice
#### (1)Please drag the perceptual.py into tha folder, and replace, or rename, your old face_rotater.py with the new face_rotater.py. You should be notice that the new face_rotater.py is different from the original one.
#### (2)You should change your batch size when you are training, because that the VGG16 requires a lot of nvram.
### 3.Ways to use perceptual loss:
#### (1)Use as class
```
  from tha.perceptual import perceptual   #that should be a class
  loss=perceptual_loss(output,target)
  loss=loss.forword()
```
#### (2)Use as function
```
  from tha.perceptual import perceptual_loss    #that should be a function
  loss=perceptual_loss(output,target)
```
### 4.Theory
#### (1)Use the VGG16 which is already modify the first layer of the model, that it can input a 4-channel picture.
#### (2)Put the output and label into the new_model, and get two outputs. Put these two outputs into MSELoss, and return it to the training function.
   
