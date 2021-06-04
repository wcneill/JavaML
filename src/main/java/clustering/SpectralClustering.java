package clustering;

import org.nd4j.linalg.api.ndarray.INDArray;


 public class SpectralClustering {

  private INDArray laplacian = null;
  private int k;


  public SpectralClustering() {
  }

  public SpectralClustering(INDArray laplacian) {
    this.laplacian = laplacian;
  }

  public void fit(INDArray data){
    //TODO calculate similarity graph, compute Laplacian
  }

  public void cluster(INDArray data, Clustering algo) {
    //TODO check for fit, if present run k-means on Laplacian
  }

}
