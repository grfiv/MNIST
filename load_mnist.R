# see https://gist.github.com/brendano/39760

# Load the MNIST digit recognition dataset into R
# http://yann.lecun.com/exdb/mnist/
# assume you have all 4 files and gunzip'd them
# creates train$n, train$x, train$y  and test$n, test$x, test$y
# e.g. train$x is a 60000 x 784 matrix, each row is one digit (28x28)
# call:  show_digit(train$x[5,])   to see a digit.
# brendan o'connor - gist.github.com/39760 - anyall.org

load_mnist <- function() {
    load_image_file <- function(filename) {
        ret = list()
        f = file(filename,'rb')
        readBin(f,'integer',n=1,size=4,endian='big')
        ret$n = readBin(f,'integer',n=1,size=4,endian='big')
        nrow = readBin(f,'integer',n=1,size=4,endian='big')
        ncol = readBin(f,'integer',n=1,size=4,endian='big')
        x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
        ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
        close(f)
        ret
    }
    load_label_file <- function(filename) {
        f = file(filename,'rb')
        readBin(f,'integer',n=1,size=4,endian='big')
        n = readBin(f,'integer',n=1,size=4,endian='big')
        y = readBin(f,'integer',n=n,size=1,signed=F)
        close(f)
        y
    }
    train <<- load_image_file('data/train-images.idx3-ubyte')
    test <<- load_image_file('data/t10k-images.idx3-ubyte')

    train$y <<- load_label_file('data/train-labels.idx1-ubyte')
    test$y <<- load_label_file('data/t10k-labels.idx1-ubyte')
}


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
