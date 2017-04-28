import java.io.File

import breeze.linalg.{DenseMatrix, _}

object Main extends App {

  val resource = getClass.getResource("boaviagem_dataset.csv").getPath
  val dataSet = csvread(new File(resource), skipLines = 1)

  val data = dataSet(::, 1 to 4)
  val y = dataSet(::, 0).toDenseMatrix.t // treat y as column vector

  val regressionModel = new LinearRegression(data, y)
  val (eTheta, costHistory) = regressionModel.gradientDescent(.4, 2501) // estimate theta by gradient
  regressionModel.plotCostFunction(costHistory)

  val normalEquation = new NormalEquation(data, y)
  val neTheta =  normalEquation.estimate // estimate theta by Normal Equation

  val house = DenseMatrix((
    1.0, // ignore
    2.0, // Numero de quartos
    50.0,  // Area Útil (m^2)
    1.0, // Vagas de Estacionamento
    0.0 // Taxa de Condomínio
  ))

  val nHouse = regressionModel.norm(house)

  println(s"R$$ ${nHouse * eTheta}, done by Gradient Descent") //something is going wrong :(
  println(s"R$$ ${house * neTheta}, done by Normal Equation") // OK :)
  println(neTheta)
  println("---------------")
  println(eTheta)
  println(s"Gradient Min Cost: ${min(costHistory)}") //something is going wrong :(
}