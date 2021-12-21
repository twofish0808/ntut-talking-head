# ntut-talking-head
### 1.Notice
#### (1)Please drag the perceptual.py into tha folder, and replace, or rename, your old face_rotater.py with the new face_rotater.py. You should be notice that the new face_rotater.py is different from the original one.
#### (2)You should reduce your batch size when you are training, because that the VGG16 requires a lot of nvram.
### 3.Ways to use perceptual loss:
#### (1)Use as class
```
  from tha.perceptual import perceptual   #that should be a class
  loss=perceptual_loss(output,target)
  loss=loss.forword()
  loss=loss+L1Loss()(output,target)
```
#### (2)Use as function
```
  from tha.perceptual import getloss    #that should be a function
  loss=getloss(output,target)
  loss=loss+L1Loss()(output,target)
```
### 4.Theory
#### (1)Use the VGG16 which is already modify the first layer of the model, that it can input a 4-channel picture.
#### (2)Put the output and label into the new_model, and get layer3, layer8, layer15 outputs,Those are three relu layers. Put each two outputs into L1Loss, and return it to the training function.

### 5.Schedule
#### 2021.08.17 由學長畢業論文交接。Start training Face_Morpher.
#### 2021.10.16 Start traing Rotator(by using L1loss). We use about 20 models by training 2 days to check whether the traing progress is OK.
#### 2021.10.29 Start traing Combiner(by using L1Loss).We use about 20 models by training 2 days to check whether the traing progress is OK.
#### 2021.11.15 Start traing Rotator(by using L1loss). Full model train.
#### 2021.12.20 Switch the Rotator to perceptual loss. With full model train.
   
