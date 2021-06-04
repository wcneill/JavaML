package clustering;


import org.knowm.xchart.*;
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

public class KMeans {

	INDArray centroids = null;
	List<List<Integer>> clusters;

	public KMeans() {
		this.clusters = new ArrayList<>();
	}

	public void fit(INDArray data, int k) {

		// Initial step (Forgy Method): randomly choose k data points as initial means.
		init(k, data);

		// Convergence criteria: Assignment step does not result in a change.
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
	 * <p>
	 * For each point in the dataset, finds the closest centroid and assigns that data point
	 * to the corresponding cluster.
	 *
	 * @param data The data to cluster.
	 * @return whether or not an update to the current clusters was made.
	 */
	private boolean assign(INDArray data) {
		INDArray currPoint;
		INDArray distances;
		boolean changed = false;

		for (int j = 0; j < data.rows(); j++) {

			currPoint = data.getRow(j).reshape(1, data.columns());

			distances = Transforms.allEuclideanDistances(currPoint, centroids, 1).mul(-1);
			int closest = distances.argMax(1).getInt();

			if (!clusters.get(closest).contains(j)) {
				clusters.get(closest).add(j);
				changed = true;
			}
		}
		return changed;
	}

	/**
	 * Performs the centroid update step of K-Means.
	 * <p>
	 * Calculates new centroids based on assignment of new points to a cluster.
	 *
	 * @param data The dataset being clustered.
	 */
	private void update(INDArray data) {

		// Iterate over the points assigned (by index) to a given cluster.
		int i = 0;
		for (List<Integer> points : clusters) {
			// get an array of indexes of points assigned to this cluster.
			int[] idxs = points.stream().mapToInt(Integer::valueOf).toArray();

			//calculate mean of this cluster's points.
			INDArray mean = data.getRows(idxs).mean(0);

			// update this cluster's centroid.
			centroids.putRow(i, mean);
			i++;
		}
	}

	public void cluster(INDArray data) {
		//TODO Check if fit was performed.
		//TODO If yes, compute correct cluster for each data point;
		//TODO update
	}

	public static void main(String[] args) {

		// --------------- Read in CSV Data -------------//
		Table df = null;
		String path = "iris.csv";

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
		Column<String> labels = (Column<String>) df.column("C4");
		df.removeColumns("C4");
		double[][] data = df.as().doubleMatrix();

		// ----------- Get independent data into Ndarray ----------//
		INDArray input = Nd4j.createFromArray(data);

		// --------------------- Run K-means ----------------------//
		KMeans km = new KMeans();
		km.fit(input, 3);

		// ------------ Append Predictions as New Column ---------//
		DoubleColumn preds = DoubleColumn.create("Predictions", df.rowCount());

		int i = 0;
		for (List<Integer> cluster : km.clusters){
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
		Table type;
		XYChart chart = new XYChartBuilder().width(1200).height(800).build();
		chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter);


		for (int j = 0; j < 3; j++){

			type = df.where(
					df.doubleColumn("Predictions")
							.isEqualTo(j)
			);

			double[] xData = type.doubleColumn(2).asDoubleArray();
			double[] yData = type.doubleColumn(3).asDoubleArray();

			String seriesName = String.format("Classification %d", j);
			chart.addSeries(seriesName, xData, yData);
		}

		new SwingWrapper<XYChart>(chart).displayChart();
	}

}