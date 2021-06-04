package clustering;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

public abstract class Clustering {

  private List<List<Integer>> clusters = new ArrayList<>();
  private int k;

  public abstract void fit();

  public abstract void cluster();



}
