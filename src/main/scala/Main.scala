package br.com.esdras

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector, csvread}

object Main extends App {

 val resource = getClass.getResource("boaviagem_dataset.csv").getPath

 val dataSet = csvread(new File(resource), skipLines = 1)

// val f = Figure()
// val p = f.subplot(0)
// p.title = "exploratory data analysis"

 val X: DenseMatrix[Double] = dataSet(::, 1 to 4)
 println(X)
// val y = dataSet(::, 0)
//println(y)
// p += plot(xs,ys, '.')
// f.refresh()

 /*
  X = [ 2 1 3; 7 1 9; 1 8 1; 3 7 4 ];
y = [2 ; 5 ; 5 ; 6];
theta_test = [0.4 ; 0.6 ; 0.8];
  */

//val X = DenseMatrix(
//  (2.0, 1.0, 3.0),
//  (7.0, 1.0, 9.0),
//  (3.0, 7.0, 4.0)
//)
//
//val y = DenseVector()
//val linear = new LinearRegression()
//linear.costFunction()
}