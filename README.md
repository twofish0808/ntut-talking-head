# ntut-talking-head
### 1.Please drag the perceptual.py into tha folder, and replace, or rename, your old face_rotater.py with the new face_rotater.py. You should be notice that the new face_rotater.py is different from the original one.
### 2.Ways to use perceptual loss:
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
   
