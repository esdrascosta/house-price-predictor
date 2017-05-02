import breeze.linalg.DenseMatrix.{horzcat, ones}
import breeze.linalg.{DenseMatrix, DenseVector, linspace, max, min}
import breeze.plot.{Figure, plot}
class LinearRegression(data: DenseMatrix[Double], y: DenseMatrix[Double]) {

  val m = data.rows
  val minValue = min(data)
  val range = max(data) - min(data)
  val X = horzcat( ones[Double](m,1) , norm(data))

  def gradientDescent(alpha: Double, numInter: Int) = {

    val costHistory = DenseVector.zeros[Double](numInter)
    var theta = DenseMatrix.zeros[Double](X.cols,1)

    for( i <- 0 until numInter) {

      theta -=  alpha/m * X.t * (X * theta - y)
      costHistory(i) = computeCost(X, y, theta) // for debug

    }
    (theta, costHistory)
  }

  def computeCost(X: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double] ) = {

    val H = X * theta - y
    val a = 1.0 / (2.0 * m)
    val cost = a *  (H.t * H)
    cost(0,0)
  }

  // Normalizes the features
  def norm(x: DenseMatrix[Double]) = (x - minValue) / range

  def plotCostFunction(costHistory: DenseVector[Double]) = {
    val xs = linspace(0, costHistory.length, costHistory.length)
    val f = Figure()
    val p = f.subplot(0)
    p.title = "Cost Function"
    p += plot(xs, costHistory, '-')
    f.saveas("costFunction.png")
  }

}