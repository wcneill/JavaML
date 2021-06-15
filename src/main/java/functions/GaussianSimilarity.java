package functions;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class GaussianSimilarity extends Function{

	private double gamma;

	public GaussianSimilarity(){
		this.gamma = 1;
	}

	public GaussianSimilarity(double gamma) {
		this.gamma = gamma;
	}

	public double compute(INDArray x, INDArray y){
		double distanceSqr = Math.pow(Transforms.euclideanDistance(x, y), 2);
		return Math.exp(-1 * gamma * distanceSqr);
	}

	public void setGamma(double g){
		this.gamma = g;
	}

}
