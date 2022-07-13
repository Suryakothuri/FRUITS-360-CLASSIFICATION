imgFolder=fullfile("fruits360dataset",'fruits-360');
imds=imageDatastore(imgFolder,"LabelSource","foldernames","IncludesubFolders",true);
AppleBraeburn = find(imds.Labels == "Apple Braeburn",1);
figure,imshow(readimage(imds,AppleBraeburn));
tbl=countEachLabel(imds);
minSetCount=min(tbl{:,2});
maxNumImages=100;
minSetCount=min(maxNumImages,minSetCount);
imds=splitEachLabel(imds,minSetCount,"randomize");
countEachLabel(imds);
net=resnet50;
deepNetworkDesigner(net);
figure,plot(net)
title("First section of ResNet50");
set(gca,"YLim",[150 170]);
net.Layers(1);
net.Layers(end);
[Training,Test]=splitEachLabel(imds,0.3,"randomize")
imageSize=net.Layers(1).InputSize
augmentedTrainingSet = augmentedImageDatastore(imageSize, Training);
augmentedTestSet = augmentedImageDatastore(imageSize, Test);
w1 = net.Layers(2).Weights;
% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')
featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
% Get training labels from the trainingSet
trainingLabels = Training.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
%Extract test features using the CNN
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Get the known labels
testLabels = Test.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))
mean(diag(confMat))


testImage = readimage(Test,53);
figure,imshow(testImage);


% Create augmentedImageDatastore to automatically resize the image when
% image features are extracted using activations.
ds = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');

% Extract image features using the CNN
imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');
%Make a prediction using the classifier.

predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')

confusionchart(testLabels, predictedLabels)

