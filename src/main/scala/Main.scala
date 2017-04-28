import java.io.File

import breeze.linalg.DenseMatrix.{horzcat, ones}
import breeze.linalg.{DenseMatrix, _}
import breeze.plot.{Figure, _}
import breeze.stats._

object Main extends App {

  val resource = getClass.getResource("boaviagem_dataset.csv").getPath
  val dataSet = csvread(new File(resource), skipLines = 1)

  val m = dataSet.rows // how many training samples
  val data = dataSet(::, 1 to 4)
  val mu =  mean(data)
  val sigma =  stddev(data)
  def norm(x: DenseMatrix[Double]) = (x - mu) / sigma

  val X = horzcat( ones[Double](m,1) , data) // By convention add ones column
  val X_norm = horzcat( ones[Double](m,1) , norm(data)) // Normalizes the features in X
  val y = dataSet(::, 0).toDenseMatrix.t // treat y as column vector

  val theta = DenseMatrix.ones[Double](X.cols,1) * mu

  val regressionModel = new LinearRegression()
  val (eTheta, costHistory) = regressionModel.gradientDescent(X_norm, y, theta, 0.4, 100)

  val xs = linspace(0, costHistory.length, costHistory.length)
  val f = Figure()
  val p = f.subplot(0)
  p.title = "Cost Function"
  p += plot(xs, costHistory, '-')
  f.refresh()

  val house = DenseMatrix((
    1.0, // ignore
    2.0, // Numero de quartos
    50.0,  // Area Útil (m^2)
    1.0, // Vagas de Estacionamento
    0.0 // Taxa de Condomínio
  ))
  val nHouse = norm(house)

  val neTheta = pinv(X.t * X) * X.t * y // doing by Normal Equation

  println(s"R$$ ${nHouse * eTheta}, done by Gradient Descent") //something is going wrong :(
  println(s"R$$ ${house * neTheta}, done by Normal Equation") // OK :)

  println(s"Gradient Min Cost: ${min(costHistory)}") //something is going wrong :(
}