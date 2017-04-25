import java.io.File

import breeze.linalg._
import breeze.plot._

object Main extends App {


 val resource = getClass.getResource("boaviagem_dataset.csv").getPath

 val dataSet = csvread(new File(resource), skipLines = 1)

 val f = Figure()
 val p = f.subplot(0)
 p.title = "exploratory data analysis"

 val xs = dataSet(::, 2)
 val ys = dataSet(::, 0)

 p += plot(xs,ys, '.')
 f.refresh()

}