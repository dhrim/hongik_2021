

# 빅데이터 분석 기반 비즈니스 개발자 양성 & 취업연계 과정

- 딥러닝을 위한 파이썬 : [python_for_deep_leaarning.md](material/python_for_deep_leaarning.md)
- 딥러닝을 위한 라이브러리 : [library_for_deep_leaarning.md](material/library_for_deep_leaarning.md)
- 딥러닝 : [deep_learning.md](material/deep_learning.md)

<br>

# 딥러닝 활용을 위한 지식 구조

```
Environment
    jupyter
	colab
	usage
		!, %, run
    GCP virtual machine
linux
	ENV
	command
		cd, pwd, ls
		mkdir, rm, cp
		head, more, tail, cat
	util
		apt
		git, wget
		grep, wc, tree
		tar, unrar, unzip
	gpu
		nvidia-smi

python
	env
		python
			interactive
			execute file
		pip
	syntax
        variable
        data
            tuple
            list
            dict
            set
        loop
        if
        comprehensive list
        function
        class
	module
		import

libray
    numpy
        load
        operation
        shape
        slicing
        reshape
        axis + sum, mean
    pandas
        load
        view
        search
        operation
            to numpy
        전처리
            결측치 처리
            이상치 처리
            카테고리 데이터 인코딩
            normalization
    matplot
        draw line graph
        scatter
        show image
    Seaborn
        histogram
        scatter
        line
        box


Deep Learning
    DNN
        concept
            layer, node, weight, bias, activation
            cost function
            GD, BP
        data
            x, y
            train, validate, test
            shuffle
        learning curve : accuracy, loss
        tuning
            overfitting, underfitting
            dropout, batch normalization, regularization
            data augmentation
        Transfer Learning
    type
        supervised
        unsupervised
        reinforcement
    model
        CNN
            vanilla, named CNN
        RNN
        GAN
    task
        Classification
        Object Detection
        Generation
	Segmentation
	Pose Extraction
	Noise Removing
	Super Resolution
	Question answering
	Auto Captioning
	Style Transfer
	Image Tranlation
    data type
    	attribute data
	image data
	natural language data
	sequence data(time serial)

TensorFlow/Keras
    basic frame
        data preparing
            x, y
            train, valid, test
            normalization
            ImageDataGenerator
        fit
        evaluate
        predict
    model
        activation function
        initializer
    tuning
        learning rate decay
        regularizer
        dropout
        batch normalization
    save/load
    compile
    	model architecture
        optimizer
        loss
        metric
```

 
