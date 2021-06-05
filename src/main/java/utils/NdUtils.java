package utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

public final class NdUtils {

    /**
     * Return the indices that would sort a 1D NDArray
     * @param a the NDArray to consider
     * @param ascending sort order
     * @return The indices of the input array, in the order that would sort the array.
     */
    public static int[] argsort(INDArray a, final boolean ascending) {
        Integer[] indexes;
        if (a.isVector()){
            indexes = new Integer[(int) a.length()];
        } else {
            indexes = new Integer[a.rows()];
        }

        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, (i1, i2) -> (ascending ? 1 : -1) * Float.compare(a.getFloat(i1), a.getFloat(i2)));
        return Arrays.stream(indexes).mapToInt(Integer::intValue).toArray();
    }

    public static INDArray sortby(INDArray input, int[] idxs) {
        input = input.getRows(idxs);
        return input;
    }

    public static void main(String[] args) {
        INDArray input = Nd4j.rand(3,3);
        System.out.println(input);
        System.out.println(input.getRows(2,1,0));

        int k = 2;
        INDArray U = input.get(NDArrayIndex.all(), NDArrayIndex.interval(0, k));
        System.out.println(U);
    }
}