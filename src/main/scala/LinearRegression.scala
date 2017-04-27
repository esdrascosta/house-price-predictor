import breeze.linalg.{DenseMatrix, DenseVector, sum}

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

      val a = alpha/m
      val H = X * _theta - y
      _theta -= a * X.t * H

      costHistory(i) = computeCost(X, y, _theta)
    }

    (_theta, costHistory)
  }

  def computeCost(X: DenseMatrix[Double],
                  y: DenseMatrix[Double],
                  theta: DenseMatrix[Double] ) = {

    val m = y.rows
    val H = X * theta - y
    val b = 1.0 / ( 2.0 * m )
    val cost: DenseMatrix[Double] = b *  H.t * H
    sum(cost)
  }
}