import breeze.linalg.{DenseMatrix, DenseVector, min, sum}

class LinearRegression {

  def gradientDescent(X: DenseMatrix[Double],
                      y: DenseMatrix[Double],
                      theta: DenseMatrix[Double],
                      alpha: Double,
                      numInter: Int): (DenseMatrix[Double], DenseVector[Double]) = {

    val costHistory = DenseVector.zeros[Double](numInter)
    val m = y.rows
    var _theta = theta

    for( i <- 0 until numInter) {

      _theta -=  alpha/m * X.t * (X * _theta - y)

      costHistory(i) = computeCost(X, y, _theta)
    }

    (_theta, costHistory)
  }

  def computeCost(X: DenseMatrix[Double],
                  y: DenseMatrix[Double],
                  theta: DenseMatrix[Double] ) = {

    val m = y.rows
    val H = X * theta - y
    val b = 1.0 / 2.0 * m
    val cost = b *  H.t * H
    cost(0,0)
  }
}