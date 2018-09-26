package mlnyan.ml;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IBackPropagation{
    INDArray calc(Layer layer, INDArray diff);

    IBackPropagation initial();

    IBackPropagation clone();
}
