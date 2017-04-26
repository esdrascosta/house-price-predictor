package br.com.esdras

import breeze.linalg.{DenseMatrix, DenseVector}

class LinearRegression {

  def gradientDescent(X: DenseMatrix[Double],
                      y: DenseVector[Double],
                      theta: DenseVector[Double],
                      alpha: Double,
                      numInteractions: Int): (DenseVector[Double], DenseVector[Double]) = {


    ???
  }

  def costFunction(X: DenseMatrix[Double],
                   y: DenseVector[Double],
                   theta: DenseVector[Double] ) = {

    val m = y.length
    val H = (X*theta - y)
    (1/(2*m)) *  (H.t * H)
  }


}
