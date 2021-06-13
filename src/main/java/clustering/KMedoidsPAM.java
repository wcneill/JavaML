package clustering;

import com.google.common.collect.Sets;
import com.google.common.primitives.Ints;
import functions.Function;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

public class KMedoidsPAM extends KClustering {

	private final int k = 10;

	private Set<Integer> X;
	private Set<Integer> medoids;
	private final Set<Set<Integer>> clusters;
	private final Function similarityFunc;

	public KMedoidsPAM(Function f) {
		this.medoids = new HashSet<>();
		this.clusters = new HashSet<>();
		this.similarityFunc = f;
	}

	@Override
	public void fit(INDArray data) {
		//TODO: Build Step
		build(data);
		//TODO: Swap Step
		swap(data);
	}

	@Override
	public List<List<Integer>> getClusters() {
		return null;
	}

	/**
	 * Initializes the first medoid of the dataset by exhaustively iterating through every
	 * candidate and finding the value that maximises similarity to all other data points.
	 *
	 * @param data The dataset that is being clustered.
	 */
	private void init(INDArray data) {
		int[] idxArr = Nd4j.arange(data.rows()).data().asInt();
		this.X = Sets.newHashSet(Ints.asList(idxArr));
		Set<Integer> candidate;

		double bestScore = Double.MAX_VALUE;
		double currScore;
		Integer bestMedoid = null;

		for (int x : X){
			currScore = 0;
			candidate = Sets.newHashSet(x);
			for (Integer x_j : Sets.difference(X, candidate)) {
				currScore += similarityFunc.compute(data.getRow(x), data.getRow(x_j));
			}
			if (currScore < bestScore) {
				bestScore = currScore;
				bestMedoid = x;
			}
		}
		medoids.add(bestMedoid);
	}

	private void build(INDArray data) {
		//TODO: Greedy init medoid l=0
		init(data);
		//TODO: For medoid_l, l=1...k
			//TODO: greedily assign next medoid.
	}

	private void swap(INDArray data) {
		//TODO: Calculate cost of medoids from build step.
		//TODO: current_loss = loss(medoids)

		//TODO: While cost is decreasing:
			//TODO: cost = swapNext()
			//TODO: if cost < last_loss
				//TODO: current_loss = cost
			//TODO: else:
				//TODO: break;
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

	private double swapNext(){
		//TODO: MX = cartesian_product(M, X - M)
		//TODO: average_loss = []

		//TODO: for each (m, x_i) in MX:
			//TODO: for each x_j in X
				//TODO: compute d = min(d(x_i, x_j), d(m_c(x_j), x_j)) --> note: second distance must exclude current medoid m
				//TODO: total_d += d;
			//TODO: average_loss[i] = total_d / len(MX)

		//TODO: swap = argmin(average_loss)
		//TODO: swapCost = average_loss[swap]

		//TODO: add x_i to M
		//TODO: remove m from M

		//TODO: return swapCost;
		return -1.0;
	}

	private int getClosestObject(INDArray x, Set<Integer> M) {
		//TODO
		return -1;
	}

}
