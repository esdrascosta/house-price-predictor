import breeze.linalg._
import breeze.math._
import breeze.numerics._
import breeze.plot.Figure
import breeze.plot._

object Main extends App {

  // TODO
  //  val resource = getClass.getResource("boaviagem_dataset.csv").getPath
  //  val dataSet = csvread(new File(resource), skipLines = 1)

  val X = DenseMatrix(
    (2.0, 1.0, 3.0),
    (7.0, 1.0, 9.0),
    (1.0, 8.0, 1.0),
    (3.0, 7.0, 4.0)
  )

  val y = DenseMatrix(
    2.0,
    5.0,
    5.0,
    6.0
  )

  val theta = DenseMatrix(
    0.0,
    0.0,
    0.0
  )

  val regressionModel = new LinearRegression()
  val (newTheta, costHistory) = regressionModel.gradientDescent(X, y, theta, 0.01, 20)

  val xs = linspace(0, costHistory.length, costHistory.length)
  val f = Figure()
  val p = f.subplot(0)
  p.title = "Cost Function"
  p += plot(xs, costHistory, '-')
  f.refresh()

}