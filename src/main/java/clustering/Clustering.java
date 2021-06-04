package clustering;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public abstract class Clustering {

  private List<List<Integer>> clusters;

  public abstract void fit(INDArray data);

  public abstract void cluster(INDArray data);

  public List<List<Integer>> getClusters(){
    return clusters;
  }

}
