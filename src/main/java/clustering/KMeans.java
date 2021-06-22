package clustering;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;
import tech.tablesaw.io.csv.CsvReadOptions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class KMeans implements KClustering {

	private int k;
	private int trials = 10;
	private INDArray centroids = null;
	private List<List<Integer>> clusters;

	public KMeans() {
		this.clusters = new ArrayList<>();
	}

	/**
	 * Runs the K-means algorithm the number of times determined by `trials`, selecting the best run
	 * based on
	 *
	 * @param data
	 */
	public void fit(INDArray data) {

		double bestVariance = Double.MAX_VALUE;
		double totalVariance;
		INDArray bestCentroids = null;

		System.out.println("Clustering: " + trials + " trials");
		for (int i = 0; i < trials; i++) {
			clusters.clear();
			totalVariance = 0;
			run(data);

			for (List<Integer> cluster : clusters) {

				int[] clusterIdxs = cluster.stream().mapToInt(Integer::valueOf).toArray();
				INDArray clusterData = data.getRows(clusterIdxs);
				double clusterVariance = clusterData.var(0).getDouble(0);

				totalVariance += clusterVariance;
			}
			System.out.println("Run " + i + " var: " + totalVariance);
			if (totalVariance < bestVariance) {
				bestVariance = totalVariance;
				bestCentroids = centroids.dup();
			}

		}
		System.out.println("Best variance: " + bestVariance);
		centroids = bestCentroids;
		cluster(data);
		System.out.println("Clustering complete. ");
	}

	/**
	 * This method is where the K-means algorithm runs.
	 *
	 * <p>Classic K-means runs in three steps: Initialize k centroids, assign data to those k centroids, update the
	 * value of those k-medoids. In this implementation, those three steps are the init, assign and update methods.
	 *
	 * @param data The data to fit K-means to.
	 */
	private void run(INDArray data) {
		init(k, data);
		boolean changed = true;
		while (changed) {
			changed = assign(data);
			update(data);
		}
	}

	/**
	 * Forgy method of initial selection of k-means.
	 *
	 * @param k    The number of means to initialize.
	 * @param data The data from which random means will be selected.
	 */
	private void init(int k, INDArray data) {
		Random r = new Random();
		int[] meanIndexes = r.ints(k, 0, data.rows()).toArray();
		centroids = data.getRows(meanIndexes);

		for (int idx : meanIndexes) {
			ArrayList<Integer> cluster = new ArrayList<>();
			cluster.add(idx);
			clusters.add(cluster);
		}
	}

	/**
	 * Perform the assignment step of K-Means clustering.
	 *
	 * <p>For each point in the dataset, finds the closest centroid and assigns that data point
	 * to the corresponding cluster.
	 *
	 * @param data The data to cluster.
	 * @return whether or not an update to the current clusters was made.
	 */
	private boolean assign(INDArray data) {
		INDArray currPoint;
		INDArray distances;
		boolean changed = false;
		int closest;

		for (int j = 0; j < data.rows(); j++) {
			currPoint = data.getRow(j).reshape(1, data.columns());
			distances = Transforms.allEuclideanDistances(currPoint, centroids, 1).mul(-1);
			closest = distances.argMax().getInt();

			if (!clusters.get(closest).contains(j)) {
				clusters.get(closest).add(j);
				changed = true;
			}
		}
		return changed;
	}

	/**
	 * Performs the centroid update step of K-Means.
	 *
	 * <p>Calculates new centroids based on assignment of new points to a cluster.
	 *
	 * @param data The dataset being clustered.
	 */
	private void update(INDArray data) {

		// Iterate over the points assigned (by index) to a given cluster.
		int i = 0;
		for (List<Integer> cluster : clusters) {
			// get an array of indexes of points assigned to this cluster.
			int[] idxs = cluster.stream().mapToInt(Integer::valueOf).toArray();

			//calculate mean of this cluster's points.
			INDArray mean = data.getRows(idxs).mean(0);

			// update this cluster's centroid.
			centroids.putRow(i, mean);
			i++;
		}
	}

	/**
	 * This method sets the value of k for the first time or updates it. If the model has
	 * already been fit, setting a new k will clear the existing clusters and the model must
	 * be re-fit.
	 *
	 * @param k the new number of clusters.
	 */
	public void setK(int k) {
		clusters.clear();
		this.k = k;
	}

	/**
	 * The number of times to run K-Means before returning the best results.
	 *
	 * @param n Number of trials.
	 */
	public void setTrials(int n) {
		this.trials = n;
	}

	/**
	 * Returns the model's current clusters based on last fit.
	 *
	 * @return A list of lists, where each of k inner lists contains the indices of the data that
	 * belongs to it.
	 */
	@Override
	public List<List<Integer>> getClusters() {
		return this.clusters;
	}

	/**
	 * Compute the nearest centroid or medoid based on pre-fit internal model.
	 *
	 * @param data The data to cluster.
	 */
	private void cluster(INDArray data) {
		for (List<Integer> cluster : clusters) {
			cluster.clear();
		}
		assign(data);
	}

	public static void main(String[] args) {

		// --------------- Read in CSV Data -------------//
		Table df = null;
		String path = "clusters_simple.csv";

		CsvReadOptions options =
			CsvReadOptions.builder(path)
				.separator(',')
				.header(false)
				.build();

		try {
			df = Table.read().usingOptions(options);
		} catch (IOException e) {
			e.printStackTrace();
		}

		// ------ Separate independent and dependent variables -----//
//		Column<String> labels = (Column<String>) df.column("C4");
//		df.removeColumns("C4");
		double[][] data = df.as().doubleMatrix();

		// ----------- Get independent data into Ndarray ----------//
		INDArray input = Nd4j.createFromArray(data);

		// --------------------- Run K-means ----------------------//
		KMeans km = new KMeans();
		km.setK(3);
		km.fit(input);

		// ------------ Append Predictions as New Column ---------//
		DoubleColumn preds = DoubleColumn.create("Predictions", df.rowCount());

		int i = 0;
		for (List<Integer> cluster : km.clusters) {
			int[] idxs = cluster.stream().mapToInt(Integer::valueOf).toArray();
			for (int ix : idxs) {
				preds.set(ix, i);
			}
			i++;
		}
		df.addColumns(preds);
		System.out.println(df.structure());
		System.out.println(df);


		// ------------------- Visualize Results -------------------- //
		Table type;
		XYChart chart = new XYChartBuilder().width(1200).height(800).build();
		chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);


		for (int j = 0; j < 3; j++) {

			type = df.where(
				df.doubleColumn("Predictions").isEqualTo(j)
			);

			double[] xData = type.doubleColumn(0).asDoubleArray();
			double[] yData = type.doubleColumn(1).asDoubleArray();

			String seriesName = String.format("Classification %d", j);
			chart.addSeries(seriesName, xData, yData);
		}
		new SwingWrapper<>(chart).displayChart();
	}
}
