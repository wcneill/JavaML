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
    //
  }

  public void cluster(INDArray data, Clustering algo) {
	//TODO check for fit, if present run k-means on Laplacian
  }

}
