import java.io.File

import breeze.linalg.DenseMatrix.{horzcat, ones}
import breeze.linalg.{DenseMatrix, _}
import breeze.plot.{Figure, _}
import breeze.stats._

object Main extends App {

  val resource = getClass.getResource("boaviagem_dataset.csv").getPath
  val dataSet = csvread(new File(resource), skipLines = 1)

  val m = dataSet.rows // how many training samples
  val X = horzcat( ones[Double](m,1) , dataSet(::, 1 to 4)) // By convention add ones column
  val mu =  mean(X)
  val sigma =  stddev(X)
  val X_norm = (X -:- mu) /:/ sigma // Normalizes the features in X
  val y = dataSet(::, 0).toDenseMatrix.t // treat y as column vector
  val theta = DenseMatrix.zeros[Double](X.cols,1)

  val regressionModel = new LinearRegression()
  val (eTheta, costHistory) = regressionModel.gradientDescent(X_norm, y, theta, 0.4, 80)

  val xs = linspace(0, costHistory.length, costHistory.length)
  val f = Figure()
  val p = f.subplot(0)
  p.title = "Cost Function"
  p += plot(xs, costHistory, '.')
  f.refresh()

  val house = DenseMatrix((
    1.0, // ignore
    2.0, // Numero de quartos
    50.0,  // Area Útil (m^2)
    1.0, // Vagas de Estacionamento
    0.0 // Taxa de Condomínio
  ))


  val norm_eq_theta = pinv(X.t *X) * X.t * y // doing by Normal Equation
  val price = house * eTheta //TODO review

  println(s"R$$ ${house * eTheta} ")
  println(s"R$$ ${house * norm_eq_theta} ")
  println("===========")
  println(norm_eq_theta)
  println("-----------")
  println(eTheta)

}