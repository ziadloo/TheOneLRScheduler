# The one learning rate scheduler to rule them all

This repo contains the Python code as well as a simple online designer to help you implement a learning rate scheduler for PyTorch. While PyTorch itself comes with a wide variety of [learning rate schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate), configuring them to generate your desired scheduler is a task on its own. Especially since an incorrecly configured scheduler could potentially waste your resources before you can identify it. That's why I believe a visual designer to configure your scheduler could go a long way.

In my search to find a visual designer for PyTorch schedulers, while I did find one, it was a few years old and I could not set it up. Also, it was not implementing all of the schedulers. On top of that, even using such an approach, you still have to find the correct values for each scheduler yourself manually. I mean such an approach definitely helps but I wanted more. That's why I came up with this design, to draw the scheduler visually using a mouse and then exporting it to Python code. It doesn't get more intuitive than this.

All you need to do is to click on the following link and it will show you how to do it (hosted on GitHub):

[The One LR Scheduler Designer](https://htmlpreview.github.io/?https://github.com/ziadloo/TheOneLRScheduler/blob/main/Designer/index.html)
