package clustering;

import functions.Function;
import functions.GaussianSimilarity;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;
import tech.tablesaw.io.csv.CsvReadOptions;
import utils.NdUtils;

import java.io.IOException;
import java.util.List;

public class SpectralClustering {

    private int k;
    private final KClustering model;
    private Function SimilarityFunction;
    private INDArray laplacian;

    /**
     * Construct a wrapper around the given KClusters model (i.e. K-means or K-medoids) that will fit
     * a laplacian matrix and pass its eigenvectors to said model for clustering.
     *
     * The default Gaussian similarity function is used for creating the affinity matrix.
     *
     * @param model The KClustering model to fit/predict with after spectral
     *              decomposition of data.
     */
    public SpectralClustering(KClustering model) {
        this.SimilarityFunction = new GaussianSimilarity();
        this.model = model;
    }

    /**
     * Construct a wrapper around the given KClusters model (i.e. K-means or K-medoids) that will fit
     * a laplacian matrix and pass its eigenvectors to said model for clustering.
     *
     * Allows for the passing of a custom similarity function.
     *
     * @param model The KClusters model to use after spectral decomposition.
     * @param f The desired similarity function used to create the affinity matrix.
     */
    public SpectralClustering(KClustering model, Function f) {
        this.model = model;
        this.SimilarityFunction = f;
    }

    /**
     * This method transforms the data into a reduced eigenspace, and then passes the reduced
     * spaces as input into the internal KClustering (K-Means or K-Medoids) model for final clustering.
     *
     * @param data The data to cluster.
     */
    public void fit(INDArray data) {

        if (k != 0) {
            //TODO calculate similarity graph, compute Laplacian
            laplacian = getLaplacian(data);

            //TODO get eigenvector matrix U from Laplacian.
            INDArray eigenvectors = laplacian.dup();
            INDArray eigenvalues = Eigen.symmetricGeneralizedEigenvalues(eigenvectors);
            eigenvectors.transposei(); // vects are now rows, for easier sorting.

            //TODO sort eigenvector matrix U according ascending eigenvalues.
            int[] sortIdx = NdUtils.argsort(eigenvalues, true);
            eigenvectors = NdUtils.sortby(eigenvectors, sortIdx);
            eigenvectors.transposei(); // back to column matrix.

            //TODO Reduce U to U' by taking only first k columns.
            INDArray uPrime = eigenvectors.get(NDArrayIndex.all(), NDArrayIndex.interval(0, k));

            //TODO pass U' to internal model fit() method.
            model.fit(uPrime);
        } else {
            System.out.println("Please use the setK() method before running fit().");
        }

    }

    /**
     * Compute the nearest centroid or medoid based on pre-fit internal model.
     *
     * @param data The data to cluster.
     */
    public List<List<Integer>> cluster(INDArray data) {
        //TODO check for fit, if present run internal model on data.
        return null;
    }

    /**
     * Calculates the Laplacian matrix L = D - A, where D is the degree matrix of the weighted
     * affinity matrix A. A itself is calculated either from the default Gaussian kernel or a custom
     * similarity function passed at construction.
     *
     * @param data The data from which to generate the Laplacian matrix.
     * @return The Laplacian matrix.
     */
    private INDArray getLaplacian(INDArray data) {
        // TODO Calculate Adjacency/Similarity Matrix A
        INDArray upperA = getUpperSimilarity(data);
        INDArray A = upperA.add(upperA.transpose());

        // TODO Calculate Diagonal Matrix D
        INDArray D = Nd4j.diag(upperA.sum(1));

        // TODO Calculates Laplacian D - A;
        return D.sub(A);
    }

    /**
     * Helper method. Computes an upper triangular matrix of similarity scores.
     *
     * @param input The data from which to generate the laplacian matrix.
     * @return Upper triangular matrix containing similarity scores .
     */
    private INDArray getUpperSimilarity(INDArray input) {

        //TODO Use similarity function to calculate Similarity Matrix A.
        INDArray similarity = Nd4j.zeros(input.rows(), input.rows());

        // For each data point (row)
        for (int i = 0; i < input.rows(); i++) {
            int j = i + 1;

            // For all other data points (rows) calculate similarity score between data points
            while (j < input.rows()) {
                double simScore = this.SimilarityFunction.compute(input.getRow(i), input.getRow(j));
                similarity.put(i, j, simScore);
                j++;
            }
        }
        return similarity;
    }

    /**
     * Returns the internal model's clusters of the fit data.
     *
     * @return A List of Lists, where each of k inner lists contains the indices of the data
     *  belonging to that cluster.
     */
    public List<List<Integer>> getClusters() {
        return model.getClusters();
    }

    /**
     * Resets the internal KCluster model's k parameter. Note that this will clear any clusters
     * that were computed under the previous k value, so fit will have to be re-run.
     *
     * @param k the k-value to pass to the internal KCluster model.
     */
    public void setK(int k) {
        this.k = k;
        this.model.setK(k);
    }

    public static void main(String[] args) {
        // --------------- Read in CSV Data -------------//
        Table df = null;
        String path = "iris.data";

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

        // --------------  Run Spectral Clustering --------------//
        KMeans km = new KMeans();
        SpectralClustering sc = new SpectralClustering(km);
        sc.setK(3);
        sc.fit(input);

        // ------------ Append Predictions as New Column ---------//
        DoubleColumn preds = DoubleColumn.create("Predictions", df.rowCount());

        int i = 0;
        for (List<Integer> cluster : sc.getClusters()) {
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


        for (int j = 0; j < 3; j++) {

            type = df.where(
                    df.doubleColumn("Predictions").isEqualTo(j)
            );

            double[] xData = type.doubleColumn(2).asDoubleArray();
            double[] yData = type.doubleColumn(3).asDoubleArray();

            String seriesName = String.format("Classification %d", j);
            chart.addSeries(seriesName, xData, yData);
        }

        new SwingWrapper<>(chart).displayChart();
    }
}
