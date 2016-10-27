lazy val commonSettings = Seq(
  version := "1.0.0",
  scalaVersion := "2.11.8"
)

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "Mxnet-Scala-DRLFlappyBird"
  )

 libraryDependencies ++= Seq(
 	"args4j" % "args4j" % "2.33",
 	"nu.pattern" % "opencv" % "2.4.9-7"
)


