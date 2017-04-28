import breeze.linalg.DenseMatrix.{horzcat, ones}
import breeze.linalg.{DenseMatrix, pinv}


class NormalEquation(data: DenseMatrix[Double], y: DenseMatrix[Double]) {

  val m = data.rows
  val X = horzcat( ones[Double](m,1) , data) // By convention add ones column

  def estimate = pinv(X.t * X) * X.t * y
}