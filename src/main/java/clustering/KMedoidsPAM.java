package clustering;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class KMedoidsPAM extends KClustering {

	private final int k = 10;
	private INDArray medoids;
	private final List<List<Integer>> clusters;

	public KMedoidsPAM() {
		this.clusters = new ArrayList<>();
	}

	@Override
	public void fit(INDArray data) {
		//TODO: Build Step
		//TODO: Swap Step

	}

	@Override
	public List<List<Integer>> getClusters() {
		return null;
	}

	private void build() {
		//TODO: Greedy init medoid l=0
		//TODO: For medoid_l, l=1...k
			//TODO: greedily assign next medoid.
	}

	private void swap() {
		//TODO: MX = cartesian_product(M, X - M)
		//TODO for each (m, x_i) in MX:
			//TODO: average_loss = []
				//TODO: for each x_j in X
					//TODO: compute d = min(d(x_i, x_j), d(m_c(x_j), x_j)) --> note: second distance must exclude current medoid m
					//TODO: total_d += d;
				//TODO: average_loss[i] = total_d / len(MX)
			//TODO: Swap = argmin(average_loss)

	}

	private int getNextMedoid(){
		//TODO: averages_loss = []
		//TODO: for x_i in X - M:
			//TODO: for x_j in X:
				//TODO: compute d = min(d(x, x_j), d(m_c(x_j), x_j))
				//TODO: total_d += d
			//TODO: average_loss[i] = total_d / len(X - M)
		//TODO: medoid_l = argmin(average_loss)
		return -1;
	}

	private int getClosestMedoid(INDArray x, Set<Integer> M) {
		//TODO
		return -1;
	}

}
