package clustering;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public abstract class KClustering {

  private int k;

  private List<List<Integer>> clusters;

  public abstract void fit(INDArray data);

  public abstract void cluster(INDArray data);

  public abstract List<List<Integer>> getClusters();

  public void setK(int k) {
    this.k = k;
  }
}
