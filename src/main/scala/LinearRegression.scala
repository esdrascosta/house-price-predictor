import breeze.linalg.DenseMatrix.{horzcat, ones}
import breeze.linalg.{DenseMatrix, DenseVector, linspace}
import breeze.plot.{Figure, plot}
import breeze.stats.{mean, stddev}

class LinearRegression(data: DenseMatrix[Double], y: DenseMatrix[Double]) {

  val m = data.rows
  val mu = mean(data)
  val sigma = stddev(data)
  val X = horzcat( ones[Double](m,1) , norm(data))


  def gradientDescent(alpha: Double, numInter: Int) = {

    val costHistory = DenseVector.zeros[Double](numInter)
    val m = y.rows
    var theta = DenseMatrix.zeros[Double](X.cols,1)

    for( i <- 0 until numInter) {

      theta -=  alpha/m * X.t * (X * theta - y)
      costHistory(i) = computeCost(X, y, theta) // for debug
    }

    (theta, costHistory)
  }

  def computeCost(X: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double] ) = {

    val H = X * theta - y
    val b = 1.0 / 2.0 * m
    val cost = b *  H.t * H
    cost(0,0)
  }

  // Normalizes the features
  def norm(x: DenseMatrix[Double]) = (x - mu) / sigma

  def plotCostFunction(costHistory: DenseVector[Double]) = {
    val xs = linspace(0, costHistory.length, costHistory.length)
    val f = Figure()
    val p = f.subplot(0)
    p.title = "Cost Function"
    p += plot(xs, costHistory, '-')
    f.refresh()
  }

}