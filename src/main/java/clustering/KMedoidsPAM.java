package clustering;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.common.collect.Table;
import com.google.common.primitives.Ints;
import functions.Function;
import functions.GaussianSimilarity;
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
	/*A similarity matrix to be computed based on input data*/
	private Table<Integer, Integer, Double> simMatrix;

	public KMedoidsPAM(Function f) {
		this.similarityFunc = f;
		this.medoids = new HashSet<>();
		this.clusters = new ArrayList<>();
		setK(8);
	}

	public KMedoidsPAM() {
		this.similarityFunc = new GaussianSimilarity();
		this.medoids = new HashSet<>();
		this.clusters = new ArrayList<>();
		setK(8);
	}

	@Override
	public void fit(INDArray data) {
		//TODO: Get similarity matrix
		init(data);
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

	private void init(INDArray data) {

		// TODO initialize set X with integer indexes of data.
		int[] idxArr = Nd4j.arange(data.rows()).data().asInt();
		this.X = Sets.newHashSet(Ints.asList(idxArr));

		//TODO build similarity matrix
		for (int i = 0; i < data.rows(); i++){
			for (int j = i + 1; j < data.rows(); j++){
				double similarity = similarityFunc.compute(data.getRow(i), data.getRow(j));
				simMatrix.put(i, j, similarity);
			}
		}
	}

	/**
	 * Initializes the first medoid of the dataset by exhaustively iterating through every
	 * candidate and finding the value that maximises similarity to all other data points.
	 */
	private void getFirstMedoid() {

		Set<Integer> candidate;

		double bestScore = Double.MAX_VALUE;
		double currScore;
		Integer bestMedoid = null;

		for (int x : X){
			currScore = 0;
			candidate = Sets.newHashSet(x);
			for (int x_j : Sets.difference(X, candidate)) {
				currScore += simMatrix.get(x, x_j);
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

		getFirstMedoid();

		for (int i = 1; i < k; i++){
			//TODO: greedily assign next medoid.
			medoids.add(getNextMedoid(data));
		}

		//TODO: assign each medoid to the cluster it represents;
		int c = 0;
		for (int m : medoids) {
			clusters.get(c).add(m);
			c++;
		}

		//TODO: assign each non-medoid to the cluster with the closest medoid;
		for (int x : Sets.difference(X, medoids)){
			int closestMedoid = getClosestMedoid(x);
			for (List<Integer> cluster : clusters) {
				if (cluster.contains(closestMedoid)) {
					cluster.add(x);
				}
			}
		}
	}

	/**
	 * A greedy search for the next medoid considering the current medoids.
	 *
	 * @param data The data from which the medoids are determined.
	 * @return the index in data of the next medoid.
	 */
	private int getNextMedoid(INDArray data){

		double runningLoss;
		double bestAverageLoss = Double.MAX_VALUE;
		int bestMedoid = -1;

		//TODO: for x_i in X - M:
		for (int i : Sets.difference(X, medoids)) {
			runningLoss = 0;

			//TODO: for x_j in X:
			for (int j : X) {
				int m = getClosestMedoid(j);
				double d1 = simMatrix.get(i, j);
				double d2 = simMatrix.get(m, j);
				double d = Math.min(d1, d2);
				runningLoss += d;
			}

			// TODO: See if the loss has decreased, update current best medoid if so.
			if (runningLoss < bestAverageLoss) {
				bestAverageLoss = runningLoss;
				bestMedoid = i;
			}
		}
		return bestMedoid;
	}

	private void swap() {

		//TODO: Calculate cost of medoids from build step.
		boolean improving = true;
		double bestLoss = getTotalAvgLoss();
		double currLoss;

		//TODO: While swapping results in improvement:



		while (improving) {
			// TODO Swap best pair
			currLoss = swapNext();
			if (currLoss < bestLoss){
				bestLoss = currLoss;
				// TODO update cluster assignments.
			} else {
				break;
			}
		}
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

			// calculate average drop in cost/dissimilarity of the current swap.
			avgLosses.put(0, xToSwap, runningLoss / X.size());
			// put the candidate medoid back.
			medoids.add(mToSwap);
		}

		int swapPairIdx = avgLosses.mul(-1).argMax().getInt();
		List<Integer> pairToSwap = MX.get(swapPairIdx);
		medoids.remove(pairToSwap.get(0));
		medoids.add(pairToSwap.get(1));

		return avgLosses.getInt(swapPairIdx);
	}

	private int getClosestMedoid(int j) {

		double bestSimilarity = Double.MAX_VALUE;
		int closestMedoid = -1;
		//TODO: get the indexes of the current medoids in the data.
		for (int m : medoids) {
			double similarity = simMatrix.get(j, m);
			if (similarity < bestSimilarity){
				bestSimilarity = similarity;
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
