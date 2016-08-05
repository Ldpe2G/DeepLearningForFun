for running this code, you need to add the following code to your Symbol object in Symbol.scala, then recompile the scala-pkg:

```scala
/**
 * Take sum of the src in the given axis
 *
 * Parameters
 * ----------
 * data : Symbol. Input data to sum.
 * axis : int, default=-1, means to reduce all the dimensions.
 */
def sumAxis(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
  createFromNamedSymbolsNoCheck("sum_axis", name, attr)
}
```
