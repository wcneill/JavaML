package clustering;

import functions.Function;
import functions.GaussianSimilarity;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


public class SpectralClustering {

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
    //TODO pass Laplacian to internal model fit() method.
  }

  public void cluster(INDArray data) {
	//TODO check for fit, if present run internal model on data.
  }

  private void getLaplacian(){
    // TODO Calculate Similarity Matrix A
    // TODO Calculate Diagonal Matrix D
    // TODO Calculates Laplacian D - A;
  }

  private INDArray getSimMatrix(){
    //TODO Use similarity function to calculate Similarity Matrix A
    return null;
  }

}
