To start a numerical evaluation of our model based on mean IoU:
sudo python2.7 eval.py --data_path ./datasets/Pascal/ --load_from_checkpoint checkpoints/unetPascal3.0/ --num_class 2 --batch_size 1000

To start a demo:
sudo python2.7 select2.0.py --num_class 2 --load_from_checkpoint checkpoints/unetPascal3.0/ --image_dir select_test/ --image_ext png
Up on start, press these keys IN ORDER:
 - S: Short of 'Select', and then you can draw a closed curve as your selection. Your stopping point will be automatically connected to the starting point. Do not make the curve too non-convex. If you want to cancel a selection and draw it once again, simply press 'S' again.
 - Space: Ensure your selection. Then the program will start to make a prediction on your input. It will take about 10 secs for the first image but very short for the proceedings.
 - N: Jump to the next image.
 - P: Back to the previous image.
 - C: Exit the demo. 

To train the network:
sudo python2.7 train1.0.py --data_path ./datasets/Pascal --checkpoint_path ./checkpoints/unetPascal3.0/ --num_class 2 --batch_size 16 --lr_decay 0.99 --epoch 100 --load_from_checkpoint checkpoints/unetPascal3.0/
If you want to train it from scratch, delete the load_from_checkpoint option.

To visualize the training procedure up on training begins:
tensorboard --logdir checkpoints/unetPascal3.0/

Introduction:
 - select2.0.py: the demo script
 - eval.py: a mean IoU evaluation script
 - train1.0.py: the training script
 - loader.py: data loader, including rules to place the dataset
 - model.py: implementation of U-Net
 - utils.py: inludes a function to calculate meanIU
 - select_test/: demo pictures
 - results/: a directory including results shown in the report
 - datasets/: dataset of VOC2012. including some scripts to transforming data and arrange them. the dataset is divided into 3 parts, test, validation, training.
    - my-VOC2012: including generated user interaction imitation: rectangles.
 - checkpoints: where to save the model weights and tensorboard event.


