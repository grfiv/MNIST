set.seed(1009)
# read MNIST training and test data
library(data.table)
library(caret)

# read the data from the csv files
# ################################
if (deskewed) {
    trainX = fread(input='../data/train-images_deskewed.csv', sep=",", header=FALSE,verbose=FALSE)
    testX  = fread(input='../data/t10k-images_deskewed.csv',  sep=",", header=FALSE,verbose=FALSE)
    print("deskewed data loaded")
} else {
    trainX = fread(input='../data/train-images.csv', sep=",", header=FALSE,verbose=FALSE)
    testX  = fread(input='../data/t10k-images.csv',  sep=",", header=FALSE,verbose=FALSE)
    print("original data loaded")
}

trainY = read.table(file='../data/train-labels.csv', sep="", header=FALSE)
testY  = read.table(file='../data/t10k-labels.csv',  sep="", header=FALSE)

trainY = as.vector(trainY$V1)
testY  = as.vector(testY$V1)

# shuffle the data to help any CV process
# #######################################
train.shuffle = sample(nrow(trainX))
trainX = trainX[train.shuffle,]
trainY = trainY[train.shuffle]

test.shuffle = sample(nrow(testX))
testX = testX[test.shuffle,]
testY = testY[test.shuffle]

rm(train.shuffle, test.shuffle)

show_digit <- function(arr784, col=gray(12:1/12), ...) {
    image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

print_16 = function(starting_at=1, X=trainX, Y=trainY) {
    # print a 4x4 of images in the training set
    # starting at index=starting_at
    opar = par(no.readonly=TRUE)
    par(mfrow=c(4,4))
    for (i in seq(from=starting_at, length.out=16)){
        show_digit(matrix(as.numeric(X[i,]),28,28),
                   main=Y[i],
                   xlab=paste("index",i))
    }
    par(opar)
}