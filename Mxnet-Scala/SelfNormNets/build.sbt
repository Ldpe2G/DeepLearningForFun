lazy val commonSettings = Seq(
  version := "1.0.0",
  scalaVersion := "2.11.8"
)

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "Mxnet-Scala-Self-Norm-Nets"
  )

libraryDependencies ++= Seq(
  "args4j" % "args4j" % "2.33",
  "org.slf4j" % "slf4j-simple" % "1.6.2" % Test,
  "org.slf4j" % "slf4j-api" % "1.6.2"
)