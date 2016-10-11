lazy val commonSettings = Seq(
  version := "1.0.0",
  scalaVersion := "2.11.8"
)

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "Mxnet-Scala-HumanActivityRecognition"
  )

 libraryDependencies += "org.sameersingh.scalaplot" % "scalaplot" % "0.0.4"


