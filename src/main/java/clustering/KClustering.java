package clustering;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

interface KClustering {

  public abstract void fit(INDArray data);

  public abstract void setK(int k);

  public abstract void setTrials(int n);

  public abstract List<List<Integer>> getClusters();


}
