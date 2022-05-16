# ntut-talking-head
## 1.The project
### 1.Face_rotator train
#### (1)Run the renamer.py, remenber to change the path in the renamer.py
#### (2)Use the new rotater and combiner to train the model, the model is the same as the old way, so you can use your own .pt file to keep training.

### 2.Schedule
#### 2021.08.17 由學長畢業論文交接。Start training Face_Morpher.
#### 2021.10.16 Start traing Rotator(by using L1loss). We use about 20 models by training 2 days to check whether the traing progress is OK.
#### 2021.10.29 Start traing Combiner(by using L1Loss).We use about 20 models by training 2 days to check whether the traing progress is OK.
#### 2021.11.15 Start traing Rotator(by using L1loss). Full model train.
#### 2021.12.20 Switch the Rotator to perceptual loss. With full model train.
#### 2021.12.26 Try to add noise on the training data.
#### 2022.03.19 Add our own app, with adjustable brightness to improve user's experience.

### 3.Run Demo
#### Replace the app folder, and put test_app folder in the root directory,and create test and checkpoint those two floders. Put your .pt file in the checkpoint folder. It should be like this.
```
	+--Talking-head-anime
	 |-test_app
	 |-test
	 |
	 +--app
	 | +-demo.py
	 | ∟poser_test.py
	 |
	 +--checkpoint
	   |-combiner.pt
	   |-face_morpher.pt
	   ∟two_algo_face_rotator.pt
```
#### And you can run like this.
```
   $python app/poser_test.py
```
#### Also the puppeteer.
##### To install enviroment
````
   $conda env create -f demo.yml
````
##### Run puppeteer.
````
   $python app/demo.py
````

### 4.enviroment
#### matplotlib(all the version is fine)
#### pytorch==1.8 or above(includes all the dependency packages)
#### python==3.6 or above(better use python 3.9.7)
#### cuda==10.2 or above (better use cuda11.1 or above)
#### We suggest you use the RTX2060 or above, or it will not be smooth in the demo.

## 2.Perceptual_loss
### 1.Notice
#### (1)Please drag the perceptual.py into tha folder, and replace, or rename, your old face_rotater.py with the new face_rotater.py. You should be notice that the new face_rotater.py is different from the original one.
#### (2)You should reduce your batch size when you are training, because that the VGG16 requires a lot of nvram.
### 2.Ways to use perceptual loss:
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
### 3.Theory
#### (1)Use the VGG16 which is already modify the first layer of the model, that it can input a 4-channel picture.
#### (2)Put the output and label into the new_model, and get layer3, layer8, layer15 outputs,Those are three relu layers. Put each two outputs into L1Loss, and return it to the training function.


   
