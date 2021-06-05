package utils;

import org.nd4j.linalg.api.ndarray.INDArray;

public class NdUtils {

 public static int[] argsort(INDArray array){
  int[] idxs = new int[array.rows()];

  for (int i = 0; i < idxs.length; i++){
   idxs[i] = i;
  }

  return null;
 }

}
