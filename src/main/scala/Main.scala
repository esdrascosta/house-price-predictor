import java.io.File

import breeze.linalg.DenseMatrix.{horzcat, ones}
import breeze.linalg.{DenseMatrix, _}

object Main extends App {

  val resource = getClass.getResource("boaviagem_dataset.csv").getPath
  val dataSet = csvread(new File(resource), skipLines = 1)

  val data = dataSet(::, 1 to 4)
  val y = dataSet(::, 0).toDenseMatrix.t // treat y as column vector

  val regressionModel = new LinearRegression(data, y)
  val (eTheta, costHistory) = regressionModel.gradientDescent(1.985, 1000) // estimate theta by gradient
  regressionModel.plotCostFunction(costHistory)

  val normalEquation = new NormalEquation(data, y)
  val neTheta =  normalEquation.estimate // estimate theta by Normal Equation

  val house = DenseMatrix((
    1.0,  // ignore
    2.0,  // Numero de quartos
    50.0, // Area Útil (m^2)
    1.0,  // Vagas de Estacionamento
    0.0   // Taxa de Condomínio
  ))

  val _house = house(::, 1 to 4) // remove ones to normalize features

  val nHouse = horzcat( ones[Double](1,1) , regressionModel.norm(_house) )

  val priceGradient = nHouse * eTheta
  val priceNormEq = house * neTheta

  println(s"R$$ $priceGradient, done by Gradient Descent")
  println(s"R$$ $priceNormEq, done by Normal Equation")
  println("---------------")
  println(s"Gradient Min Cost: ${min(costHistory)}") // something is going wrong :(
}