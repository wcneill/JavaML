package clustering;

import functions.Function;
import functions.GaussianSimilarity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndexAll;

import java.util.List;


public class SpectralClustering {

  private int k;
  private Clustering model;
  private Function SimilarityFunction;
  private INDArray laplacian;
  private List<List<Integer>> clusters;

  public SpectralClustering(Clustering model) {
    this.SimilarityFunction = new GaussianSimilarity();
    this.model = model;
  }

  public SpectralClustering(Clustering model, Function f) {
    this.model = model;
    this.SimilarityFunction = f;
  }

  public void fit(INDArray data){
	//TODO calculate similarity graph, compute Laplacian
    laplacian = getLaplacian(data);

    //TODO get eigenvector matrix U from Laplacian.
    INDArray eigenvectors = laplacian.dup();
    INDArray eigenvalues = Eigen.symmetricGeneralizedEigenvalues(eigenvectors);

    //TODO sort eigenvector matrix U according ascending eigenvalues.
    //eigenvalues.argSort

    //TODO Reduce U to U' by taking only first k columns.
    INDArray uPrime = eigenvectors.get(NDArrayIndex.all(), NDArrayIndex.interval(0, k));

    //TODO pass U' to internal model fit() method.
    model.fit(uPrime);
    this.clusters = model.getClusters();

  }

  public void cluster(INDArray data) {
	//TODO check for fit, if present run internal model on data.
  }

  private INDArray getLaplacian(INDArray data){
    // TODO Calculate Adjacency/Similarity Matrix A
    INDArray upperA = getUpperSimilarity(data);
    INDArray A = upperA.add(upperA.transpose());

    // TODO Calculate Diagonal Matrix D
    INDArray D = Nd4j.diag(upperA.sum(1));

    // TODO Calculates Laplacian D - A;
    return D.sub(A);
  }

  private INDArray getUpperSimilarity(INDArray input){
    //TODO Use similarity function to calculate Similarity Matrix A.
    //TODO Optimize this?
    INDArray similarity = Nd4j.zeros(input.rows());

    // For each data point (row)
    for (int i = 0; i < input.rows(); i++) {
      int j = i + 1;

      // For all other data points (rows)
      while (j < input.rows()){

        // calculate similarity score between data points
        double simScore = this.SimilarityFunction.compute(input.getRow(i), input.getRow(j));
        similarity.put(i, j, simScore);
      }
    }
    return similarity;
  }
}
