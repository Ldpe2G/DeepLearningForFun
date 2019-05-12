lazy val commonSettings = Seq(
  version := "1.0.0",
  scalaVersion := "2.11.8"
)

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "Mxnet-Scala-FastNeuralStyle"
  )

libraryDependencies ++= Seq(
	"com.sksamuel.scrimage" % "scrimage-core_2.11" % "2.1.7",
	"com.sksamuel.scrimage" % "scrimage-filters_2.11" % "2.1.7",
	"com.sksamuel.scrimage" % "scrimage-io-extra_2.11" % "2.1.7",
	"args4j" % "args4j" % "2.33",
      "org.slf4j" % "slf4j-simple" % "1.6.2" % Test,
      "org.slf4j" % "slf4j-api" % "1.6.2",
      "nu.pattern" % "opencv" % "2.4.9-7"
)


