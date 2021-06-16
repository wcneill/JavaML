package clustering;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.common.primitives.Ints;
import functions.Function;
import org.apache.commons.compress.utils.Lists;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

public class KMedoidsPAM implements KClustering {

	/* The number of clusters. Default number of clusters is 8. use the setK method to change this value.*/
	private int k;
	/* The set of integer values representing the the data by inded */
	private Set<Integer> X;
	/* The set of indexes representing the medoids within the data */
	private Set<Integer> medoids;
	/* The clusters within the data.
	Each internal list contains the indexes corresponding to the data that belongs to it.*/
	private final List<List<Integer>> clusters;
	/* The similarity function used to fit the data to a Laplacian matrix. */
	private final Function similarityFunc;

	public KMedoidsPAM(Function f) {
		this.similarityFunc = f;
		this.medoids = new HashSet<>();
		this.clusters = new ArrayList<>();
		setK(8);
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
		return clusters;
	}

	@Override
	public void setK(int k) {
		clusters.clear();
		this.k = k;
		for (int i = 0; i < k; i ++){
			clusters.add(new ArrayList<Integer>());
		}
	}

	@Override
	public void setTrials(int n) {
		// To satisfy interface?
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

	/**
	 * The build step of the PAM algorithm. Greedily assigns k-medoids.
	 *
	 * @param data The data to cluster.
	 */
	private void build(INDArray data) {
		//TODO: Greedy init medoid l=0
		init(data);
		//TODO: For medoid_l, l=1...k
		for (int i = 1; i < k; i++){
			//TODO: greedily assign next medoid.
			medoids.add(getNextMedoid(data));
		}

		//TODO: assign each medoid to the cluster it represents;
		for (int m : medoids) {
			clusters.get(m).add(m);
		}

		//TODO: assign each data point to the cluster with the closest medoid;
		for (int i : X){
			int m = getClosestMedoid(i, data);
			clusters.get(m).add(i);
		}
	}

	private void swap(INDArray data) {
		//TODO: Calculate cost of medoids from build step.

		boolean improving = true;
		double bestLoss = getTotalAvgLoss(data);
		double currLoss;

		while (improving) {
			currLoss = swapNext();
		}

		//TODO: While cost is decreasing:
			//TODO: cost = swapNext()
			//TODO: if cost < last_loss
				//TODO: current_loss = cost
			//TODO: else:
				//TODO: break;
	}

	/**
	 * A greedy search for the next medoid considering the current medoids.
	 *
	 * @param data The data from which the medoids are determined.
	 * @return the index in data of the next medoid.
	 */
	private int getNextMedoid(INDArray data){

		int searchSize = X.size() - medoids.size();
		INDArray avgLosses = Nd4j.create(searchSize);
		double runningLoss;

		//TODO: for x_i in X - M:
		for (int i : Sets.difference(X, medoids)) {
			runningLoss = 0;

			//TODO: for x_j in X:
			for (int j : X) {

				// The distances d2 could be pre-computed and cached to make this operation O(kn^2) instead of O(k^2n^2)
				int m = getClosestMedoid(j, data);
				double d1 = similarityFunc.compute(data.getRow(i), data.getRow(j));
				double d2 = similarityFunc.compute(data.getRow(m), data.getRow(j));
				double d = Math.min(d1, d2);
				runningLoss += d;
			}
			avgLosses.put(0, i, runningLoss / X.size());
		}
		return (avgLosses.mul(-1)).argMax().getInt();
	}

	private double swapNext(INDArray data){
		// Get the cartesian product, that is all pairs of candidates (x, m) to swap.
		// Then, convert to array list because we cannot index into a set, but we need to later.
		Set<List<Integer>> product = Sets.cartesianProduct(medoids, Sets.difference(X, medoids));
		ArrayList<List<Integer>> MX = Lists.newArrayList(product.iterator());
		INDArray avgLosses = Nd4j.create(MX.size());
		double runningLoss;

		//TODO: for each (m, x_i) in MX:
		for (List<Integer> pair : MX) {
			int mToSwap = pair.get(0);
			int xToSwap = pair.get(1);
			runningLoss = 0;

			// temporarily remove candidate medoid in order to test swap with non-medoid candidate.
			medoids.remove(mToSwap);

			// calculate the reduction in cost of this swap.
			for (int j : X) {
				int m = getClosestMedoid(j, data);
				double d1 = similarityFunc.compute(data.getRow(xToSwap), data.getRow(j));
				double d2 = similarityFunc.compute(data.getRow(m), data.getRow(j));
				double d = Math.min(d1, d2);
				runningLoss += d;
			}

			// calculate average drop in cost of the current swap.
			avgLosses.put(0, xToSwap, runningLoss / X.size());
			// put the candidate medoid back.
			medoids.add(mToSwap);
		}

		int swapPairIdx = avgLosses.mul(-1).argMax().getInt();
		List<Integer> pairToSwap = MX.get(swapPairIdx);
		medoids.remove(pairToSwap.get(0));
		medoids.add(pairToSwap.get(1));


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

	private int getClosestMedoid(int j, INDArray data) {

		// TODO: get the indexes of the current medoids in the data.
		int[] idxs = Ints.toArray(this.medoids);

		// TODO: get the current datapoint for which to find the closest medoid
		INDArray object = data.getRow(j);

		double similarity;
		double bestSim = Double.MAX_VALUE;
		int closestMedoid = -1;

		// TODO: Find the shortest distance from point j to any medoid.
		for (int m : idxs) {
			similarity = similarityFunc.compute(object, data.getRow(m));
			if (similarity < bestSim){
				bestSim = similarity;
				closestMedoid = m;
			}
		}
		return closestMedoid;
	}

	/**
	 * Finds the total average distance between each cluster's medoid and the points belonging to those clusters.
	 *
	 * @param data the data being clustered.
	 * @return The sum of similarities of all points to their medoids, divided by the number of points.
	 */
	private double getTotalAvgLoss(INDArray data) {
		double similaritySum = 0;
		for (int m : medoids) {
			// indexes of all the data points belonging to this cluster.
			int[] idxs = clusters.get(m).stream().mapToInt(Integer::intValue).toArray();
			INDArray medoidObject = data.getRow(m).reshape(1, data.columns());

			//Calculate the sum of similarities between each point and this cluster's medoid.
			for (int x : idxs) {
				similaritySum += similarityFunc.compute(data.getRow(x), medoidObject);
			}
		}
		return similaritySum / k;
	}

	public static void main(String[] args) {
		INDArray a = Nd4j.createFromArray(new double[]{1, 1, 1}).reshape(1, 3);
		INDArray b = Nd4j.createFromArray(new double[]{2, 2, 2}).reshape(1, 3);
		INDArray c = Nd4j.createFromArray(new double[]{3, 3, 3}).reshape(1, 3);
		c = Nd4j.concat(0, b, c);
		System.out.println(c);

		INDArray distances = Transforms.allEuclideanDistances(a, c, 1);
		System.out.println(distances);

		System.out.println("hooo aaaa!");
		System.out.println(distances);
		System.out.println(distances.sumNumber());
		System.out.println(distances.meanNumber());
		double n = 0;
		n += (double) distances.sumNumber();
		System.out.println(n);
	}

}
