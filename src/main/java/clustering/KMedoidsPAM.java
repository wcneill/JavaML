package clustering;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Sets;
import com.google.common.collect.Table;
import com.google.common.primitives.Ints;
import functions.Function;
import functions.GaussianSimilarity;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.columns.Column;
import tech.tablesaw.io.csv.CsvReadOptions;

import java.io.IOException;
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
		this.simMatrix = HashBasedTable.create();
		setK(8);
	}

	public KMedoidsPAM() {
		this.similarityFunc = new GaussianSimilarity();
		this.medoids = new HashSet<>();
		this.clusters = new ArrayList<>();
		this.simMatrix = HashBasedTable.create();
		setK(8);
	}

	@Override
	public void fit(INDArray data) {
		//TODO: Get similarity matrix
		init(data);
		//TODO: Build Step
		build(data);
		//TODO: Swap Step
		swap();

		for (int x : Sets.difference(X, medoids)){
			int closestMedoid = getClosestMedoid(x);
			for (List<Integer> cluster : clusters) {
				if (cluster.contains(closestMedoid)) {
					cluster.add(x);
				}
			}
		}
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
			simMatrix.put(i, i, -1 * similarityFunc.compute(data.getRow(i), data.getRow(i)));
			for (int j = i + 1; j < data.rows(); j++){
				double similarity = similarityFunc.compute(data.getRow(i), data.getRow(j));
				simMatrix.put(i, j, -1 * similarity);
				simMatrix.put(j, i, -1 * similarity);
			}
		}
	}

	/**
	 * The build step of the PAM algorithm. Greedily assigns k-medoids.
	 *
	 */
	private void build(INDArray data) {

		getFirstMedoid();

		for (int i = 1; i < k; i++){
			//TODO: greedily assign next medoid.
			medoids.add(getNextMedoid());
		}

		//TODO: assign each medoid to the cluster it represents;
		int c = 0;
		for (int m : medoids) {
			clusters.get(c).add(m);
			c++;
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
	 * A greedy search for the next medoid considering the current medoids.
	 *
	 * @return the index in data of the next medoid.
	 */
	private int getNextMedoid(){

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
			if ((runningLoss / X.size()) < bestAverageLoss) {
				bestAverageLoss = runningLoss / X.size();
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
		currLoss = swapNext(bestLoss);
		while (currLoss < bestLoss) {
			bestLoss = currLoss;
			currLoss = swapNext(bestLoss);
		}
	}

	private double swapNext(double bestAverageLoss){
		// Get all pairs of candidates (x, m) to swap.
		Set<List<Integer>> mCrossX = Sets.cartesianProduct(medoids, Sets.difference(X, medoids));
		double runningLoss;
		List<Integer> bestPair = null;

		//TODO: for each (m, x_i) candidate swap in in M x X/M:
		for (List<Integer> pair : mCrossX) {

			runningLoss = 0;
			int mCandidate = pair.get(0);
			int xCandidate = pair.get(1);
			medoids.remove(mCandidate);

			// calculate effect of this swap.
			for (int j : X) {
				int m = getClosestMedoid(j);
				double d1 = simMatrix.get(xCandidate, j);
				double d2 = simMatrix.get(m, j);
				double d = Math.min(d1, d2);
				runningLoss += d;
			}

			medoids.add(mCandidate);

			if (runningLoss / X.size() < bestAverageLoss){
				bestAverageLoss = runningLoss / X.size();
				bestPair = pair;
			}
		}
		if (bestPair != null) {
			medoids.remove(bestPair.get(0));
			medoids.add(bestPair.get(1));
		}
		return bestAverageLoss;
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
	 * @return The sum of similarities of all points to their medoids, divided by the number of points.
	 */
	private double getTotalAvgLoss() {
		double similaritySum = 0;
		for (int x : Sets.difference(X, medoids)) {
			int m = getClosestMedoid(x);
			similaritySum += simMatrix.get(m, x);
		}
		return similaritySum / k;
	}

	public static void main(String[] args) {
		// --------------- Read in CSV Data -------------//
		tech.tablesaw.api.Table df = null;
		String path = "iris.data";

		CsvReadOptions options =
			CsvReadOptions.builder(path)
				.separator(',')
				.header(false)
				.build();

		try {
			df = tech.tablesaw.api.Table.read().usingOptions(options);
		} catch (IOException e) {
			e.printStackTrace();
		}

		// ------ Separate independent and dependent variables -----//
		Column<String> labels = (Column<String>) df.column("C4");
		df.removeColumns("C4");
		double[][] data = df.as().doubleMatrix();

		// ----------- Get independent data into Ndarray ----------//
		INDArray input = Nd4j.createFromArray(data);

		// --------------------- Run K-means ----------------------//
		KMedoidsPAM pam = new KMedoidsPAM();
		pam.setK(3);
		pam.fit(input);

		// ------------ Append Predictions as New Column ---------//
		DoubleColumn preds = DoubleColumn.create("Predictions", df.rowCount());

		int i = 0;
		for (List<Integer> cluster : pam.clusters) {
			int[] idxs = cluster.stream().mapToInt(Integer::valueOf).toArray();
			for (int ix : idxs) {
				preds.set(ix, i);
			}
			i++;
		}
		df.addColumns(labels, preds);
		System.out.println(df.structure());
		System.out.println(df);


		// ------------------- Visualize Results -------------------- //
		tech.tablesaw.api.Table type;
		XYChart chart = new XYChartBuilder().width(1200).height(800).build();
		chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);


		for (int j = 0; j < 3; j++) {

			type = df.where(
				df.doubleColumn("Predictions").isEqualTo(j)
			);

			double[] xData = type.doubleColumn(2).asDoubleArray();
			double[] yData = type.doubleColumn(3).asDoubleArray();

			String seriesName = String.format("Classification %d", j);
			chart.addSeries(seriesName, xData, yData);
		}

		int[] idxs = Ints.toArray(pam.medoids);
		INDArray medoids = input.getRows(idxs).transpose();
		double[] x = medoids.getRow(2).toDoubleVector();
		double[] y = medoids.getRow(3).toDoubleVector();
		chart.addSeries("Medoids", x, y);
		new SwingWrapper<>(chart).displayChart();
	}
}
