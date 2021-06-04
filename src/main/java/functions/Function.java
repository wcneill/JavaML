package functions;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Function {

	public abstract double compute(INDArray x, INDArray y);

}
